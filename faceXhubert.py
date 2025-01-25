import torch
import torch.nn as nn
from hubert.modeling_hubert import HubertModel
import torch.nn.functional as F


def inputRepresentationAdjustment(audio_embedding_matrix, vertex_matrix, ifps, ofps):
    if ifps % ofps == 0:
        factor = -1 * (-ifps // ofps)
        if audio_embedding_matrix.shape[1] % 2 != 0:
            audio_embedding_matrix = audio_embedding_matrix[:, :audio_embedding_matrix.shape[1] - 1]

        if audio_embedding_matrix.shape[1] > vertex_matrix.shape[1] * 2:
            audio_embedding_matrix = audio_embedding_matrix[:, :vertex_matrix.shape[1] * 2]

        elif audio_embedding_matrix.shape[1] < vertex_matrix.shape[1] * 2:
            vertex_matrix = vertex_matrix[:, :audio_embedding_matrix.shape[1] // 2]
    else:
        factor = -1 * (-ifps // ofps)
        audio_embedding_seq_len = vertex_matrix.shape[1] * factor
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)
        audio_embedding_matrix = F.interpolate(audio_embedding_matrix, size=audio_embedding_seq_len, align_corners=True, mode='linear')
        audio_embedding_matrix = audio_embedding_matrix.transpose(1, 2)

    frame_num = vertex_matrix.shape[1]
    audio_embedding_matrix = torch.reshape(audio_embedding_matrix, (1, audio_embedding_matrix.shape[1] // factor, audio_embedding_matrix.shape[2] * factor))

    return audio_embedding_matrix, vertex_matrix, frame_num


class FaceXHuBERT(nn.Module):
    def __init__(self, args):
        super(FaceXHuBERT, self).__init__()
        """
        - audio: (batch_size, raw_wav)
        - template: (batch_size, V*3)
        - vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        self.input_fps = args.input_fps
        self.output_fps = args.output_fps
        self.gru_layers = 2
        self.gru_hidden_dim = args.feature_dim 

        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_feature_dim = self.audio_encoder.encoder.config.hidden_size
        self.audio_encoder.feature_extractor._freeze_parameters()

        frozen_layers = [0, 1]
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith("feature_projection"):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer_idx = int(name.split(".")[2])
                if layer_idx in frozen_layers:
                    param.requires_grad = False

        self.gru = nn.GRU(
            input_size=self.audio_feature_dim * 2, 
            hidden_size=self.gru_hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(self.gru_hidden_dim, args.vertice_dim)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

        num_subjects = len(args.train_subjects.split())
        self.subject_embedding = nn.Linear(num_subjects, self.gru_hidden_dim, bias=False)

    def forward(self, audio_input, template_vertices, target_vertices, one_hot_subject, loss_fn):
        template_vertices = template_vertices.unsqueeze(1)  # (batch_size, 1, V*3)
        subject_embedding = self.subject_embedding(one_hot_subject)  # (batch_size, feature_dim)
        audio_features = self.audio_encoder(audio_input).last_hidden_state  # (batch_size, seq_len, audio_feature_dim)
        audio_features, target_vertices, num_frames = inputRepresentationAdjustment(
            audio_features, target_vertices, self.input_fps, self.output_fps
        )
        audio_features = audio_features[:, :num_frames] 
        batch_size = audio_features.size(0)
        initial_hidden_state = torch.zeros(
            self.gru_layers, batch_size, self.gru_hidden_dim
        ).requires_grad_().to(audio_features.device)

        gru_output, _ = self.gru(audio_features, initial_hidden_state)  # (batch_size, seq_len, hidden_dim)

        gru_output = gru_output * subject_embedding.unsqueeze(1)

        predicted_vertices = self.fc(gru_output)  # (batch_size, seq_len, V*3)
        predicted_vertices = predicted_vertices + template_vertices

        loss = loss_fn(predicted_vertices, target_vertices)
        return torch.mean(loss)

    def predict(self, audio_input, template_vertices, one_hot_subject):
        """
        - audio_input: (batch_size, raw_wav)
        - template_vertices: (batch_size, V*3)
        - one_hot_subject: (batch_size, num_subjects)
        - predicted_vertices: (batch_size, seq_len, V*3)
        """
        template_vertices = template_vertices.unsqueeze(1)  # (batch_size, 1, V*3)
        subject_embedding = self.subject_embedding(one_hot_subject)  # (batch_size, feature_dim)

        audio_features = self.audio_encoder(audio_input).last_hidden_state

        if audio_features.size(1) % 2 != 0:
            audio_features = audio_features[:, :-1]
        audio_features = audio_features.view(1, audio_features.size(1) // 2, audio_features.size(2) * 2)

        batch_size = audio_features.size(0)
        initial_hidden_state = torch.zeros(
            self.gru_layers, batch_size, self.gru_hidden_dim
        ).requires_grad_().to(audio_features.device)
        gru_output, _ = self.gru(audio_features, initial_hidden_state)

        gru_output = gru_output * subject_embedding.unsqueeze(1)

        predicted_vertices = self.fc(gru_output)  # (batch_size, seq_len, V*3)
        predicted_vertices = predicted_vertices + template_vertices 

        return predicted_vertices

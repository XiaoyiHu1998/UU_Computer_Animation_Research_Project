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
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset_name = args.dataset
        self.input_frame_rate = args.input_fps
        self.output_frame_rate = args.output_fps
        self.num_gru_layers = 2
        self.hidden_state_dim = args.feature_dim 

        self.audio_feature_extractor = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_embedding_dim = self.audio_feature_extractor.encoder.config.hidden_size
        self.audio_feature_extractor.feature_extractor._freeze_parameters()

        frozen_layers_indices = [0, 1]
        for param_name, param_value in self.audio_feature_extractor.named_parameters():
            if param_name.startswith("feature_projection"):
                param_value.requires_grad = False
            if param_name.startswith("encoder.layers"):
                layer_number = int(param_name.split(".")[2])
                if layer_number in frozen_layers_indices:
                    param_value.requires_grad = False

        self.gru_module = nn.GRU(
            input_size=self.audio_embedding_dim * 2, 
            hidden_size=self.hidden_state_dim,
            num_layers=self.num_gru_layers,
            batch_first=True,
            dropout=0.3
        )

        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_state_dim, num_heads=4, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_state_dim, args.vertice_dim)
        nn.init.constant_(self.output_layer.weight, 0)
        nn.init.constant_(self.output_layer.bias, 0)

        subject_count = len(args.train_subjects.split())
        self.subject_embedding_layer = nn.Linear(subject_count, self.hidden_state_dim, bias=False)

    def forward(self, audio_input, base_template, target_sequence, subject_one_hot, loss_function):
        base_template = base_template.unsqueeze(1)  # (batch_size, 1, V*3)
        subject_features = self.subject_embedding_layer(subject_one_hot)  # (batch_size, feature_dim)
        extracted_audio_features = self.audio_feature_extractor(audio_input).last_hidden_state  # (batch_size, seq_len, audio_feature_dim)
        
        adjusted_audio, adjusted_targets, frame_count = inputRepresentationAdjustment(
            extracted_audio_features, target_sequence, self.input_frame_rate, self.output_frame_rate
        )
        adjusted_audio = adjusted_audio[:, :frame_count] 
        batch_count = adjusted_audio.size(0)
        
        initial_states = torch.zeros(
            self.num_gru_layers, batch_count, self.hidden_state_dim
        ).requires_grad_().to(adjusted_audio.device)

        rnn_output, _ = self.gru_module(adjusted_audio, initial_states)  # (batch_size, seq_len, hidden_dim)

        rnn_output = rnn_output * subject_features.unsqueeze(1)

        output_vertices = self.output_layer(rnn_output)  # (batch_size, seq_len, V*3)
        output_vertices = output_vertices + base_template

        computed_loss = loss_function(output_vertices, target_sequence)
        return torch.mean(computed_loss)

    def predict(self, audio_input, base_template, subject_one_hot):
        """
        - audio_input: (batch_size, raw_wav)
        - base_template: (batch_size, V*3)
        - subject_one_hot: (batch_size, num_subjects)
        - predicted_sequence: (batch_size, seq_len, V*3)
        """
        base_template = base_template.unsqueeze(1)  # (batch_size, 1, V*3)
        subject_features = self.subject_embedding_layer(subject_one_hot)  # (batch_size, feature_dim)

        extracted_audio_features = self.audio_feature_extractor(audio_input).last_hidden_state

        if extracted_audio_features.size(1) % 2 != 0:
            extracted_audio_features = extracted_audio_features[:, :-1]
        reshaped_audio = extracted_audio_features.view(
            1, extracted_audio_features.size(1) // 2, extracted_audio_features.size(2) * 2
        )

        batch_count = reshaped_audio.size(0)
        initial_states = torch.zeros(
            self.num_gru_layers, batch_count, self.hidden_state_dim
        ).requires_grad_().to(reshaped_audio.device)

        rnn_output, _ = self.gru_module(reshaped_audio, initial_states)

        rnn_output = rnn_output * subject_features.unsqueeze(1)

        predicted_sequence = self.output_layer(rnn_output)  # (batch_size, seq_len, V*3)
        predicted_sequence = predicted_sequence + base_template 

        return predicted_sequence

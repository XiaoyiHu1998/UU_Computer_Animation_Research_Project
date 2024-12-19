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

        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_dim = self.audio_encoder.config.hidden_size  # 768

        self.gru_layer_dim = 2
        self.gru_hidden_dim = args.feature_dim
        self.decoder_gru = nn.GRU(self.gru_hidden_dim, self.gru_hidden_dim, self.gru_layer_dim, batch_first=True, dropout=0.3)

        self.fc = nn.Linear(self.gru_hidden_dim, args.vertice_dim)

        self.vertice_to_hidden = nn.Linear(args.vertice_dim, self.gru_hidden_dim)

        self.attention = nn.Linear(self.gru_hidden_dim * 2, self.audio_dim)
        self.context_layer = nn.Linear(self.audio_dim, self.gru_hidden_dim)

    def forward(self, audio, template, vertice, one_hot, criterion, use_teacher_forcing=True):
        encoder_outputs = self.audio_encoder(audio).last_hidden_state
        batch_size, seq_len, _ = encoder_outputs.shape

        min_frame_num = min(seq_len, vertice.shape[1])
        encoder_outputs = encoder_outputs[:, :min_frame_num, :]
        vertice = vertice[:, :min_frame_num, :]

        decoder_hidden = torch.zeros(self.gru_layer_dim, batch_size, self.gru_hidden_dim).to(audio.device)
        current_vertice = template.unsqueeze(1)

        predictions = []
        loss = 0.0

        for t in range(min_frame_num):
            context_vector = encoder_outputs[:, t:t+1, :] 

            vertice_feature = self.vertice_to_hidden(current_vertice.squeeze(1))

            attention_input = torch.cat([vertice_feature, decoder_hidden[-1]], dim=-1)
            context_vector = self.attention(attention_input)
            context_vector = self.context_layer(context_vector.unsqueeze(1))

            decoder_input = context_vector
            output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)  

            vertice_pred = self.fc(output.squeeze(1))

            loss += criterion(vertice_pred, vertice[:, t, :])

            if use_teacher_forcing and torch.rand(1).item() < 0.5:
                current_vertice = vertice[:, t:t+1, :]
            else:
                current_vertice = vertice_pred.unsqueeze(1)

            predictions.append(vertice_pred.unsqueeze(1))

        predictions = torch.cat(predictions, dim=1)
        loss = loss / min_frame_num
        return predictions, loss


    def predict(self, audio, template, one_hot):
        encoder_outputs = self.audio_encoder(audio).last_hidden_state
        batch_size, seq_len, _ = encoder_outputs.shape
    
        decoder_hidden = torch.zeros(self.gru_layer_dim, batch_size, self.gru_hidden_dim).to(audio.device)
        current_vertice = template.unsqueeze(1)
    
        predictions = []
    
        for t in range(seq_len):
            if hasattr(self, "attention"):
                context_vector = self.attention(torch.cat([current_vertice.squeeze(1), decoder_hidden[-1]], dim=-1))
                context_vector = self.context_layer(context_vector.unsqueeze(1))
            else:
                context_vector = encoder_outputs[:, t:t+1, :]
    
            decoder_input = context_vector
    
            output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            vertice_pred = self.fc(output.squeeze(1))
            current_vertice = vertice_pred.unsqueeze(1)
    
            predictions.append(vertice_pred.unsqueeze(1))
    
        predictions = torch.cat(predictions, dim=1)
        return predictions

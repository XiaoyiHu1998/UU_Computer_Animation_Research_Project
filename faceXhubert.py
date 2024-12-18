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
        
        self.i_fps = args.input_fps  
        self.o_fps = args.output_fps  

        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.audio_dim = self.audio_encoder.config.hidden_size

        self.gru_layer_dim = 2
        self.gru_hidden_dim = args.feature_dim
        self.gru = nn.GRU(self.audio_dim + self.audio_dim, self.gru_hidden_dim, self.gru_layer_dim, batch_first=True, dropout=0.3)

        self.fc = nn.Linear(self.gru_hidden_dim, args.vertice_dim)

        self.vertice_dim_reducer = nn.Linear(args.vertice_dim, self.audio_dim)

        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)


    def forward(self, audio, template, vertice, one_hot, criterion, use_teacher_forcing=True):
        template = template.unsqueeze(1)
        current_vertice = self.vertice_dim_reducer(template)
    
        hidden_states = self.audio_encoder(audio).last_hidden_state
        frame_num = min(hidden_states.shape[1], vertice.shape[1])
    
        if frame_num == 0:
            raise ValueError("Frame_num is zero after adjustment. Check input data.")
    
        hidden_states = hidden_states[:, :frame_num, :]
        vertice = vertice[:, :frame_num, :]
    
        h0 = torch.zeros(self.gru_layer_dim, hidden_states.shape[0], self.gru_hidden_dim).to(audio.device)
        vertice_out = []
        loss = 0.0
    
        for t in range(frame_num):
            current_audio_feature = hidden_states[:, t:t+1, :]
            gru_input = torch.cat([current_audio_feature, current_vertice], dim=-1)
    
            vertice_pred, h0 = self.gru(gru_input, h0)
            vertice_pred = self.fc(vertice_pred.squeeze(1))
    
            loss += criterion(vertice_pred, vertice[:, t, :])
    
            if use_teacher_forcing and torch.rand(1).item() < 0.5:
                current_vertice = self.vertice_dim_reducer(vertice[:, t:t+1, :])
            else:
                current_vertice = self.vertice_dim_reducer(vertice_pred.unsqueeze(1))
    
            vertice_out.append(vertice_pred.unsqueeze(1))
    
        vertice_out = torch.cat(vertice_out, dim=1)
        loss = loss / frame_num
        return vertice_out, loss


    def predict(self, audio, template, one_hot):
        template = template.unsqueeze(1)  
        obj_embedding = self.obj_vector(one_hot) 
        current_vertice = self.vertice_dim_reducer(template) 
    
        hidden_states = self.audio_encoder(audio).last_hidden_state
        frame_num = hidden_states.shape[1]
        h0 = torch.zeros(self.gru_layer_dim, hidden_states.shape[0], self.gru_hidden_dim).to(audio.device)
    
        vertice_out = []
    
        for t in range(frame_num):
            current_audio_feature = hidden_states[:, t:t+1, :]
            gru_input = torch.cat([current_audio_feature, current_vertice], dim=-1)
            vertice_pred, h0 = self.gru(gru_input, h0)
            vertice_pred = self.fc(vertice_pred.squeeze(1))
            current_vertice = self.vertice_dim_reducer(vertice_pred.unsqueeze(1))
            vertice_out.append(vertice_pred.unsqueeze(1))
    
        vertice_out = torch.cat(vertice_out, dim=1)
        return vertice_out

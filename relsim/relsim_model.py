import torch
import torch.nn as nn


class QwenWithQueryToken(nn.Module):
    def __init__(self, base_model, processor, hidden_size=3584):
        """
        Wraps Qwen model and adds a learnable query token as a special token in vocabulary.
        The query token goes through the entire LLM backbone.
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.processor = processor
        
        # Add special query token to vocabulary
        special_token = "<|query|>"
        num_added = processor.tokenizer.add_special_tokens(
            {'additional_special_tokens': [special_token]}
        )
        
        if num_added > 0:
            # Resize token embeddings
            base_model.resize_token_embeddings(len(processor.tokenizer))
            
            # Get the token id for our query token
            self.query_token_id = processor.tokenizer.convert_tokens_to_ids(special_token)
            print(f"Added query token '{special_token}' with id {self.query_token_id}")
        else:
            self.query_token_id = processor.tokenizer.convert_tokens_to_ids(special_token)
            print(f"⚠️ Query token '{special_token}' already exists with id {self.query_token_id}")
        
        # Projection head to map to sentence transformer dimension (384 for all-MiniLM-L6-v2)
        self.projection = nn.Linear(hidden_size, 384)
        
        # Move projection to same device and dtype as base_model
        try:
            param = next(base_model.parameters())
            self.projection = self.projection.to(device=param.device, dtype=param.dtype)
            print(f"Projection head moved to device={param.device}, dtype={param.dtype}")
        except StopIteration:
            pass  # Model has no parameters, leave projection on CPU/float32
        
    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        """
        Forward pass that:
        1. Processes input through Qwen (query token is already in input_ids)
        2. Finds the position of query token
        3. Extracts feature from query token position
        """
        # Get outputs from base model with output_hidden_states
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get last hidden state: [batch_size, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1]
        
        # Find query token positions in each sequence
        batch_size = input_ids.shape[0]
        query_features_list = []
        
        for i in range(batch_size):
            # Find where query token is in this sequence
            query_positions = (input_ids[i] == self.query_token_id).nonzero(as_tuple=True)[0]
            
            # Strict check: must have exactly one query token
            if len(query_positions) == 0:
                raise ValueError(
                    f"Query token (id={self.query_token_id}) not found in batch sample {i}. "
                    f"This indicates the input was not properly preprocessed."
                )
            elif len(query_positions) > 1:
                raise ValueError(
                    f"Multiple query tokens found in batch sample {i} at positions {query_positions.tolist()}. "
                    f"Expected exactly one query token per sample."
                )
            
            # Extract feature from the single query token position
            query_pos = query_positions[0].item()
            query_features_list.append(hidden_states[i, query_pos, :])
        
        # Stack to get [batch_size, hidden_size]
        query_features = torch.stack(query_features_list, dim=0)
        
        # Project to sentence transformer dimension
        projected_features = self.projection(query_features)  # [batch_size, 384]
        
        return projected_features


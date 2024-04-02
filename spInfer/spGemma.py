from typing import Optional, Tuple
import torch
from transformers.models import gemma

from spInfer.utils import ColorPrinter


class spGemmaForCausalLM(gemma.GemmaForCausalLM):
    def __init__(
        self,
        *args,
        guess_length: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._guess_length = guess_length
        self._guess_model: Optional[gemma.GemmaForCausalLM] = None
        self._cprint = ColorPrinter()

    def set_guess_model(
        self,
        model: torch.nn.Module,
        pad_token_id: int = 0,
    ) -> None:
        """Sets the guess model"""
        self._guess_model = model
        self._pad_token_id = pad_token_id

    def _guess(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates guesses from the guess model"""
        if max_new_tokens is None:
            max_new_tokens = self._guess_length
        attn_mask_ext = (
            attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        )

        input_ids_ext = input_ids.clone()
        guesses = self._guess_model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)

        # Pad first rows of input_ids_ext and attn_mask_ext to length of guesses
        input_ids_ext = torch.cat([torch.tensor([self._pad_token_id]*(len(guesses[0])-len(input_ids[0]))).unsqueeze(0), input_ids], dim=1)
        attn_mask_ext = torch.cat([torch.tensor([0]*(len(guesses[0])-len(attn_mask_ext[0]))).unsqueeze(0), attn_mask_ext], dim=1)

        for i in range(len(input_ids[0]), len(guesses[0])):
            next_input = torch.cat([input_ids_ext[-1][1:], guesses[0][i].unsqueeze(0)], dim=0)
            input_ids_ext = torch.cat([input_ids_ext, next_input.unsqueeze(0)], dim=0)
            next_attn_mask = torch.cat([attn_mask_ext[-1][1:], torch.tensor([1])], dim=0)
            attn_mask_ext = torch.cat([attn_mask_ext, next_attn_mask.unsqueeze(0)], dim=0)

        return input_ids_ext, attn_mask_ext
    
    def generate_with_guess(
        self, input_ids: torch.Tensor, max_new_tokens: int = 30, attention_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Uses speculative inference technique to generate sequences."""
        assert self._guess_model is not None, "Please set the guess model using `set_guess_model()`."
        
        max_output_length = 0
        output_list = []

        for b_i in range(len(input_ids)):
            current_input_ids = input_ids[b_i].clone().detach().unsqueeze(0)
            current_attention_mask = attention_mask[b_i].clone().unsqueeze(0) if attention_mask is not None else None
            remaining_new_tokens = max_new_tokens
            self._cprint(current_input_ids.tolist(), end=' ')

            while remaining_new_tokens > 0:
                if remaining_new_tokens > 1:
                    l_guess = min(self._guess_length, remaining_new_tokens-1)
                    input_ids_ext, attn_mask_ext = self._guess(current_input_ids, max_new_tokens=l_guess, attention_mask=current_attention_mask)
                else:   # Skip guessing when only 1 token left
                    l_guess = 0
                    input_ids_ext, attn_mask_ext = current_attention_mask, current_attention_mask

                speculative_output_ids = self.generate(input_ids_ext, max_new_tokens=1, attention_mask=attn_mask_ext, **kwargs)
                i = 0
                while i < len(input_ids_ext)-1:
                    if input_ids_ext[i+1][-1] != speculative_output_ids[i][-1]:
                        break
                    i += 1
                n_pads = len(input_ids_ext)-i-1
                new_ids = speculative_output_ids[i][n_pads:][len(current_input_ids[0]):]
                self._cprint(new_ids.tolist(), end=' ')
                current_input_ids = speculative_output_ids[i][n_pads:].unsqueeze(0)
                current_attention_mask = torch.cat([attn_mask_ext[i][n_pads:], torch.tensor([1])]).unsqueeze(0)
                remaining_new_tokens -= i+1
            output_list.append(current_input_ids[0].tolist())
            max_output_length = max(max_output_length, len(output_list[-1]))
            self._cprint.reset()
            print()
        for i in range(len(output_list)):
            output_list[i] = [self._pad_token_id]*(max_output_length-len(output_list[i])) + output_list[i]
        return torch.tensor(output_list)

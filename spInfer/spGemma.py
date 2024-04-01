import torch
from transformers.models import gemma
from colorama import Fore, Style


class spGemmaForCausalLM(gemma.GemmaForCausalLM):
    colors = [Fore.BLUE, Fore.CYAN, Fore.GREEN, Fore.MAGENTA, Fore.RED, Fore.WHITE, Fore.YELLOW]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guess_model = None
        self._guess_length = 5

    def set_guess_model(self, model: gemma.GemmaForCausalLM, pad_token_id=0):
        self._guess_model = model
        self._pad_token_id = pad_token_id

    @property
    def guess_model(self):
        return self._guess_model

    def generate_guess(self, input_ids, max_new_tokens=None, attention_mask=None, **kwargs):
        if max_new_tokens is None:
            max_new_tokens = self._guess_length
        if attention_mask is None:
            extended_attention_mask = torch.ones_like(input_ids).tolist()
        else:
            extended_attention_mask = attention_mask.clone().tolist()

        extended_input_ids = input_ids.clone().tolist()
        guesses = self._guess_model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)

        for i in range(len(input_ids[0]), len(guesses[0])):
            extended_input_ids.append(extended_input_ids[-1]+[guesses[0][i].item()])
            extended_attention_mask.append(extended_attention_mask[-1]+[1])
        for i in range(len(extended_input_ids)):
            extended_input_ids[i] = [self._pad_token_id]*(len(extended_input_ids[-1])-len(extended_input_ids[i])) + extended_input_ids[i]
            extended_attention_mask[i] = [0]*(len(extended_attention_mask[-1])-len(extended_attention_mask[i])) + extended_attention_mask[i]
        extended_input_ids = torch.tensor(extended_input_ids)
        extended_attention_mask = torch.tensor(extended_attention_mask)

        return extended_input_ids, extended_attention_mask
    
    def generate_with_guess(self, input_ids, max_new_tokens=30, attention_mask=None, **kwargs):
        output_ids = input_ids.clone().tolist()
        output_attention_mask = attention_mask.clone().tolist() if attention_mask is not None else [None]*len(input_ids)
        max_output_length = 0

        for b_i in range(len(output_ids)):
            current_input_ids = torch.tensor(output_ids[b_i]).reshape(1, -1)
            current_attention_mask = torch.tensor([output_attention_mask[b_i]]) if output_attention_mask[b_i] is not None else None
            remaining_new_tokens = max_new_tokens
            # print(current_input_ids, current_attention_mask)
            print(self.colors[0] + f" {output_ids[b_i]}", end='')
            self.colors = self.colors[1:] + [self.colors[0]]

            while remaining_new_tokens > 0:
                if remaining_new_tokens > 1:
                    l_guess = min(self._guess_length, remaining_new_tokens-1)
                    extended_input_ids, extended_attention_mask = self.generate_guess(current_input_ids, max_new_tokens=l_guess, attention_mask=current_attention_mask)
                else:   # current_input_ids.shape[1]-output_ids[b_i].shape[1] == max_new_tokens-1
                    l_guess = 0
                    extended_input_ids, extended_attention_mask = current_attention_mask, current_attention_mask

                suspective_output_ids = self.generate(extended_input_ids, max_new_tokens=1, attention_mask=extended_attention_mask, **kwargs)
                i = 0
                while i < len(extended_input_ids)-1:
                    if extended_input_ids[i+1][-1] != suspective_output_ids[i][-1]:
                        # print(f"Skipping until: {i}\n{extended_input_ids[i+1]}\n{suspective_output_ids[i]}\n")
                        break
                    i += 1
                n_pads = len(extended_input_ids)-i-1
                new_ids = suspective_output_ids[i][n_pads:][len(current_input_ids[0]):]
                print(self.colors[0] + f" {new_ids.tolist()}", end='')
                self.colors = self.colors[1:] + [self.colors[0]]
                current_input_ids = suspective_output_ids[i][n_pads:].reshape(1, -1)
                current_attention_mask = torch.cat([extended_attention_mask[i][n_pads:], torch.tensor([1])]).reshape(1, -1)
                remaining_new_tokens -= i+1
                # print(f"{current_input_ids=}\n{current_attention_mask=}\n{remaining_new_tokens=}\n")
            output_ids[b_i] = current_input_ids[0].tolist()
            max_output_length = max(max_output_length, len(output_ids[b_i]))
            print(Style.RESET_ALL)
        for i in range(len(output_ids)):
            output_ids[i] = [self._pad_token_id]*(max_output_length-len(output_ids[i])) + output_ids[i]
        return torch.tensor(output_ids)

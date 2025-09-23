import torch


def generate(model, tokenizers, input_text, max_length, device):
    model.eval()
    src_input_ids = tokenizers["en"].encode(input_text, return_tensors="pt").to(device)
    src_padding_mask = src_input_ids != tokenizers["en"].pad_token_id.to(device)

    input_length = src_input_ids.size(1)

    if input_length >= max_length:
        raise ValueError("Input text is too long")

    tgt_input_ids = torch.tensor([[tokenizers["cy"].bos_token_id]], device=device)
    tgt_padding_mask = tgt_input_ids != tokenizers["cy"].pad_token_id.to(device)
    with torch.no_grad():
        for _ in range(max_length):
            src_enc = model.encode(src_input_ids, src_padding_mask)
            tgt_dec = model.decode(src_enc, tgt_input_ids, src_padding_mask, tgt_padding_mask)
            logits = model.output(tgt_dec)
            pred_token_id = logits[:, -1, :].argmax(dim=-1)
            tgt_input_ids = torch.cat([tgt_input_ids, pred_token_id.unsqueeze(-1)], dim=-1)
            tgt_padding_mask = tgt_input_ids != tokenizers["cy"].pad_token_id.to(device)
            if pred_token_id.item() == tokenizers["cy"].eos_token_id:
                break

    generated_ids = tgt_input_ids
    generated_text = tokenizers["cy"].decode(
        generated_ids.squeeze().tolist(), skip_special_tokens=True
    )
    return generated_text

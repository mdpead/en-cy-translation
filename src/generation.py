import torch



def generate_texts(model, tokenizers, input_texts, max_length, device):
    model = model.to(device)
    model.eval()
    src_input_ids = tokenizers["en"](input_texts, return_tensors="pt")["input_ids"].to(device)
    src_padding_mask = (src_input_ids != tokenizers["en"].pad_token_id).to(device)

    input_length = src_input_ids.size(1)

    if input_length >= max_length:
        raise ValueError("Input text is too long")

    tgt_input_ids = torch.tensor([tokenizers["cy"].bos_token_id] * src_input_ids.size(0), device=device).unsqueeze(-1)
    tgt_padding_mask = (tgt_input_ids != tokenizers["cy"].pad_token_id).to(device)
    with torch.no_grad():
        for _ in range(max_length - 1):
            src_enc = model.encode(src_input_ids, src_padding_mask)
            tgt_dec = model.decode(src_enc, tgt_input_ids, src_padding_mask, tgt_padding_mask)
            logits = model.output(tgt_dec)
            pred_token_ids = logits[:, -1, :].argmax(dim=-1)
            tgt_input_ids = torch.cat([tgt_input_ids, pred_token_ids.unsqueeze(-1)], dim=-1)
            tgt_padding_mask = (tgt_input_ids != tokenizers["cy"].pad_token_id).to(device)
            if torch.all(torch.any(tgt_input_ids == tokenizers["cy"].eos_token_id, dim=-1)):
                break

    generated_ids = tgt_input_ids
    generated_text = tokenizers["cy"].batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text

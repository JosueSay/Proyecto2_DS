import torch

def collateFn(batch):
    # batch es lista de (encA, encB, y, tie_dummy, meta)
    encA_list, encB_list, y_list, tie_list, meta_list = zip(*batch)

    def stackDict(lst):
        keys = lst[0].keys()
        return {k: torch.stack([ex[k] for ex in lst], dim=0) for k in keys}

    encA = stackDict(encA_list)
    encB = stackDict(encB_list)
    y = torch.stack(y_list)
    tie_dummy = torch.stack(tie_list)

    # longitudes efectivas usando attention_mask
    def maskLens(enc):
        return enc["attention_mask"].sum(dim=1)

    lensA = maskLens(encA)
    lensB = maskLens(encB)
    pair_lens = torch.maximum(lensA, lensB)
    input_ids_len_max = int(pair_lens.max().item())
    input_ids_len_mean = float(pair_lens.float().mean().item())

    # bandera de truncado si se toca el tope
    seq_max_cfg = int(meta_list[0]["seq_max"])
    max_len_prompt_cfg = int(meta_list[0]["max_len_prompt_cfg"])
    max_len_resp_cfg   = int(meta_list[0]["max_len_resp_cfg"])
    truncated_batch = int((pair_lens >= seq_max_cfg).any().item())

    batch_meta = {
        "input_ids_len_max": input_ids_len_max,
        "input_ids_len_mean": input_ids_len_mean,
        "truncated_batch": truncated_batch,
        "max_len_prompt_cfg": max_len_prompt_cfg,
        "max_len_resp_cfg": max_len_resp_cfg,
        "total_budget_cfg": seq_max_cfg,
    }
    return encA, encB, y, tie_dummy, batch_meta

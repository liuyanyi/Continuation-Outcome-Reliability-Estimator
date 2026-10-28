from transformers import AutoModelForCausalLM, AutoTokenizer

from cognitive_probe.probe_model.probe_model import Qwen3ForProbing, SingleProbeHead

if __name__ == "__main__":
    model_name_or_path = "outputs/diag_matrix/exp5_mlp_segment_mean_logit_features"
    model = Qwen3ForProbing.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model.export_score_head("outputs/diag_matrix/exp5_head_only")


    head_model = SingleProbeHead.from_pretrained("outputs/diag_matrix/exp5_head_only")
    print(head_model)
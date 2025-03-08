from transformers.trainer import Trainer 
import torch 
import torch.nn as nn
import json

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def fix_grad_nan_inf(model):
    layers = model.model.layers
    count = 0 
    total_params = 0
    for m in model.parameters():
    	if m.requires_grad:
    		if torch.isnan(m.grad).any() or torch.isinf(m.grad).any():
    			m.grad.zero_()


def mask_grad(model):
    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            mask = (W==0)
            subset[name].weight.grad[mask]= 0
 
def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        # print(f"layer {i} sparsity {float(sub_count)/sub_params:.4f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 
def kldiv_loss(student_logits, teacher_logits, temperature=1):
    "Kullback-Leibler divergence loss"
    num_tokens = student_logits.numel() / student_logits.size(-1)
    return (
        nn.functional.kl_div(
            input=nn.functional.log_softmax(student_logits / temperature, dim=-1),
            target=nn.functional.log_softmax(teacher_logits / temperature, dim=-1),
            log_target=True,
            reduction="sum",
        )
        * (temperature**2)
        / num_tokens
    )
class SparseTrainer(Trainer):
    def __init__(self, model= None, args= None, data_collator= None, train_dataset= None, eval_dataset= None, 
            tokenizer= None, model_init= None, compute_metrics= None, callbacks= None, optimizers= (None, None),
            preprocess_logits_for_metrics= None, teacher= None
            ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, 
                                optimizers, preprocess_logits_for_metrics)
        self.counter = 0
        self.teacher = teacher
    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.

        access optimizer through: self.optimizer.optimizer.param_groups[0] 
        """
        self.counter += 1
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        self.accelerator.backward(loss)
        # if super().do_grad_scaling: ## False 
        #     self.scaler.scale(loss).backward()
        # elif super().use_apex:   ## False 
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     self.accelerator.backward(loss)
        #     # pass 

        mask_grad(model)   ### mask the gradients

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.

        ## model type: transformers.models.llama.modeling_llama.LlamaForCausalLM
        ## outputs[0]: a single scalar
        ## outputs[1]: shape (bs, 2048, 32000)

        ## inputs["input_ids"] shape: (bs, 2048)
        ## inputs["attention_mask] shape: (bs, 2048)
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if self.args.loss_type == "SquareHead":
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
            loss_gen_tokens = inputs['labels'] != -100
            student_logits = outputs.logits[loss_gen_tokens]
            teacher_logits = teacher_outputs.logits[loss_gen_tokens]
            squarehead_loss = torch.tensor(0.0, device=outputs.logits.device, dtype=outputs.logits.dtype)
            layerwise_losses = []
            for i in range(1, len(outputs.hidden_states)):
                useful_tokens = inputs['attention_mask'] == 1
                student_states = outputs.hidden_states[i][useful_tokens]
                teacher_states = teacher_outputs.hidden_states[i][useful_tokens]
                layerwise_losses.append((student_states - teacher_states).pow(2).mean() / (teacher_states.pow(2).mean() + torch.finfo(outputs.logits.dtype).eps))

            squarehead_loss = sum(layerwise_losses)
            # print(squarehead_loss)
            # useful_tokens = inputs['attention_mask'] == 1
            # eps = torch.finfo(outputs.logits.dtype).eps  

            # student_states = [s[useful_tokens] for s in outputs.hidden_states[1:]]
            # teacher_states = [t[useful_tokens] for t in teacher_outputs.hidden_states[1:]]
            
            # teacher_norm = [torch.max(t.pow(2).mean(), eps) for t in teacher_states]
            # layerwise_losses = [(s - t).pow(2).mean() / tn for s, t, tn in zip(student_states, teacher_states, teacher_norm)]
            # squarehead_loss = sum(layerwise_losses)
            # print(squarehead_loss)
            loss += squarehead_loss
        elif self.args.loss_type == "KLDiv":
            with torch.no_grad():
                teacher_outputs = self.teacher(**inputs)
            loss_gen_tokens = inputs['labels'] != -100
            student_logits = outputs.logits[loss_gen_tokens]
            teacher_logits = teacher_outputs.logits[loss_gen_tokens]

            kl_loss = kldiv_loss(student_logits, teacher_logits)
            loss += kl_loss
        

        return (loss, outputs) if return_outputs else loss
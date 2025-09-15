
model_infer = EffnetAneurysmClassifier(
    model_name="efficientnet_b0",
    num_classes=NUM_LABELS,
    pretrained=False,
    return_logits=False   # directly output probs
).cuda()
model_infer.load_state_dict(torch.load("best.pt"))
model_infer.eval()

with torch.no_grad():
    vols, labels, metas = next(iter(train_loader))
    probs = model_infer(vols.cuda())  # [B, 15], each in [0,1]

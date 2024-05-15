import opinion_aen

model_path = '~/opinion_aen/state_dict/aen_bert_CCF_val_acc0.9048'
model = opinion_aen.model(model_path)
inputs = opinion_aen.Input(data).data  # do some input preprocessing
prob, polar = model.predict(inputs)
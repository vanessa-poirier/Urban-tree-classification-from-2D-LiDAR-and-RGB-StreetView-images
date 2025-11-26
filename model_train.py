# THIS IS A SHORT AND SWEET SCRIPT FOR MODEL TRAINING
# to be done after configuration.py
model = train_model(model, train_loader, val_loader, criterion, optimizer, device) # function train model is defined in training_pipeline.py

## EXAMPLE USE FOR SAVING
root_dir = 'E:/dir'
model_save_path = os.path.join(root_dir, "ResNet_results", "model_name.pth")
torch.save(model.state_dict(), model_save_path) 

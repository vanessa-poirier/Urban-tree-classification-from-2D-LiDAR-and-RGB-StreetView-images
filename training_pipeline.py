# TRAINING PIPELINE FOR THE RESNET TREE CLASSIFICATION MODELS
# first with a single data source (i.e. 2D-LiDAR or Streetview images), named train_model_single
# and second with dual data sources (i.e. 2D-LiDAR AND Streetview images), named train_model_dual

# Import necessary libraries
# NONE

def train_model(model, train_loader, val_loader, criterion, optimizer, device, type="dual",
                num_epochs=20, lr_decay=0.75, decay_every=5, patience=5):
    """
        Args:
            model: Trained pytorch model to be evaluated
            train_loader: Data Loader object containing the training data
            val_loader: Data Loader object containing the validation data
            criterion: Criterion used for model training
            optimizer: Optimizer used for model training
            device: CPU or GPU used for model training
            type: Character specifying whether the the model was trained with a single data source or with two data sources. Accepted values are "single" and "dual"
            num_epochs: Number of training epochs
            lr_decay: Learning rate decay
            decay_every: Number of epochs to run before reducing the learning rate
            patience: Number of epochs to wait before observing a minimal improvement in model accuracy
    """
                  
    print(device)
    best_model_wts = None
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        if epoch % decay_every == 0 and epoch > 0:
            for g in optimizer.param_groups:
                g["lr"] *= lr_decay
            print(f"[Epoch {epoch}] ➤ Learning rate decayed.")

        # TRAIN
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if type == "single":
            for inputs, labels in train_loader:
               #print("loading training data")
                inputs = inputs.to(device)  # (B, N, C, H, W)
                labels = labels.to(device)

                optimizer.zero_grad()
                #print("about to train model")
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

        elif type == "dual":
            for inputs1, inputs2, labels in train_loader:
               #print("loading training data")
                inputs1 = inputs1.to(device)  # (B, N, C, H, W)
                inputs2 = inputs2.to(device)  
                labels = labels.to(device)

                optimizer.zero_grad()
                #print("about to train model")
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        if type == "single":
            with torch.no_grad():
                for inputs, labels in val_loader:
                    #print("loading validation data")
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    #print("about to put validation data into model")
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

        elif type == "dual":
            with torch.no_grad():
                for inputs1, inputs2, labels in val_loader:
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs1, inputs2)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs1.size(0)
                    preds = outputs.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} ➤ Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")

        # Early stopping
        # Not needed if you decrease the learning rate each N epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping.")
                break

    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model

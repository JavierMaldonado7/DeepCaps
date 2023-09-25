import torch


def train(model, train_loader, test_loader, device, epochs, optimizer, criterion, save_path):
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f}")
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        # Evaluate on test set
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {epoch_loss:.4f} - Test Acc: {epoch_acc:.4f}")
        test_loss_history.append(epoch_loss)
        test_acc_history.append(epoch_acc)

        # Save the model if it has the best accuracy so far
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1}")

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history

import torch


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, recon = model(data)

            loss = criterion(data, output, target, recon)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += data.size(0)

    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total

    print('Test set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss, test_correct, test_total, test_acc))

    return test_loss, test_acc

# -----------------------------
# Evaluation on test set
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        outputs = model(X_batch)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

#Classify one new window


model.eval()

new_window = simulate_selection_window(40, 100)   # or your real data window
new_tensor = torch.tensor(new_window, dtype=torch.float32)

# add channel and batch dimensions
new_tensor = new_tensor.unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(new_tensor)
    prob = torch.sigmoid(output)
    pred = (prob > 0.5).float()

print("Probability of selection:", prob.item())
print("Predicted class:", int(pred.item()))
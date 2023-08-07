import pandas as pd
import torch
from classifier import vectorizer, classifier

data = pd.read_csv('data.csv')
data = data.sample(frac=1, ignore_index=True)
data['url_vector'] = data['url'].apply(vectorizer).apply(lambda x: x.view(1, -1))
data['label_vector'] = data['label'].replace({'bad': 1, 'good': 0}).apply(lambda x: torch.tensor(x).view(1, 1))

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)

for epoch in range(200):
    for i in range(0, len(data), 5):
        batch_urls = torch.cat(data['url_vector'][i:i+5].tolist())
        batch_labels = torch.cat(data['label_vector'][i:i+5].tolist()).float()
        optimizer.zero_grad()
        outputs = classifier(batch_urls)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch + 1}\nLoss: {loss}\n')
    torch.save(classifier.state_dict(), 'classifier.pth')
    print('Saved Classifier')

torch.save(classifier.state_dict(), 'classifier.pth')
print('Saved Classifier')
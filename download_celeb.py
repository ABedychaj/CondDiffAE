import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# Set your desired download directory
data_root = './data/celeba'

# Define any image transforms you need
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Download the CelebA dataset
celeba_dataset = torchvision.datasets.CelebA(
    root=data_root,
    split='all',           # options: 'train', 'valid', 'test', 'all'
    target_type='attr',    # get attribute labels
    transform=transform
)

# Example: Create a DataLoader for batch processing
dataloader = DataLoader(celeba_dataset, batch_size=32, shuffle=True)

print(f"Downloaded CelebA dataset to {data_root}")
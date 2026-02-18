from src.data.dataset_loader import create_splits

train_df, test_df = create_splits()

print(train_df.head())

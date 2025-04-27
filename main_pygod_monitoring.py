from torch_geometric.data import Data   
from main_pygod import *
import matplotlib
matplotlib.use('Agg')
def main(config_path=None):

    config = load_config(config_path)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_out = config["training"].get("save_dir", "./saved_models")
    out_dir = os.path.join(base_out, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"→ Outputs will be saved to {out_dir}")

    train_dataset, train_input_nodes = load_dataset(
        mask="train",
        use_aggregated=config["data"]["use_aggregated"],
        use_temporal=config["data"]["use_temporal"],
    )
    train_loader = make_loader(
        data=train_dataset,
        loader_type="neighbor",
        batch_size=config["data"]["batch_size"],
        input_nodes=train_input_nodes,
    )
    train_data = get_data_from_loader(train_loader)
    train_data = transform_data(
        train_data,
        perturb=config["transform"]["perturb"],
        interpolate=config["transform"]["interpolate"],
    )

    test_dataset, test_input_nodes = load_dataset(
        mask="test",
        use_aggregated=config["data"]["use_aggregated"],
        use_temporal=config["data"]["use_temporal"],
    )

    test_loader = make_loader(
        data=test_dataset,
        loader_type="neighbor",
        batch_size=config["data"]["batch_size"],
        input_nodes=test_input_nodes,
    )

    test_data = get_data_from_loader(test_loader)

    model = create_model(config=config["model"])
    trained_model = train_model(
        model,
        train_data,
        eval_data=test_data,
        output_directory=out_dir,
        save_embeddings=config["training"]["save_embeddings"],
        timestamp=timestamp
    )

    return trained_model

def train_model(
        model:DOMINANT,
        data:Data,
        eval_data:Data=None,
        device='cpu',
        output_directory="./outputs",
        save_embeddings=False,
        timestamp:str=None) -> DOMINANT:
    print("training model...")
    os.makedirs(output_directory, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    data = data   

    # Train the DOMINANT model
    with torch.set_grad_enabled(True):  # Explicitly control gradient tracking
        model.fit(data, eval_data=eval_data)
    print("model training complete!")
    del data

    if save_embeddings:
        embeddings = model.emb.detach().cpu().numpy()
        output_path = os.path.join(output_directory, f'embeddings_{timestamp}.npy')
        np.save(output_path, embeddings)
        print(f"embeddings saved to {output_path}")

        labels = model.label_.detach().cpu().numpy()
        labels_path = os.path.join(output_directory, f"labels_{timestamp}.npy")
        np.save(labels_path, labels)
        print(f"labels saved to {labels_path}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate DOMINANT model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the YAML configuration file')
    args = parser.parse_args()
    main(config_path=args.config)

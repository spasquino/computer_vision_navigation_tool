from src import preprocess, train, predict
import os

def main():
    print("Step 1: Preprocessing (frame extraction)")
    source_folder = os.path.join("data", "raw")  # Assumes videos/images are already here
    target_folder = os.path.join("data", "raw")  # Overwrites or appends in-place
    preprocess.extract_frames_from_all_videos(source_folder, target_folder)

    print("Step 2: Training")
    train.train_model()

    print("Step 3: Predicting a sample image")
    # Replace this with a real image path after training
    sample_image = os.path.join("data", "raw", "some_folder", "image.jpg")
    if os.path.exists(sample_image):
        predict.predict_image(sample_image)
    else:
        print(f"Sample image '{sample_image}' not found, skipping prediction.")

if __name__ == "__main__":
    main()


    print("\nStep 4: Foundation Model Embedding with CLIP")
    from src.foundation import get_clip_embedding
    # Replace with actual image path
    sample_image = os.path.join("data", "raw", "some_folder", "image.jpg")
    if os.path.exists(sample_image):
        embedding = get_clip_embedding(sample_image)
        print("CLIP embedding shape:", embedding.shape)
    else:
        print("No image found for embedding.")

    print("\nStep 5: Route Optimization")
    from src.navigation import build_route_graph, shortest_path, plot_graph
    points = ["A", "B", "C", "D"]
    connections = [("A", "B", 1), ("B", "C", 2), ("A", "C", 4), ("C", "D", 1)]
    G = build_route_graph(points, connections)
    path = shortest_path(G, "A", "D")
    print("Shortest path from A to D:", path)
    plot_graph(G, path)

import torch
from torch.utils.data import TensorDataset, DataLoader
from evaluation import calculate_prediction_metrics

def feature_erasure(
    model, 
    ranked_features, 
    test_dataloader, 
    feature_names, 
    top_n, 
    batch_size, 
    logger,
    name,
    is_multiclass=False # Added so you can pass it to the final function
):
    logger.log_text(f"Starting Feature Erasure {name} for the top-{top_n} features...")
    
    # 1. Identify the features to be zeroed and their indices in the tensor
    features_to_zero = ranked_features[:top_n]
    
    # Map the feature name to its position (index) in the Dataloader
    indices_to_zero = [
        feature_names.index(feat) 
        for feat in features_to_zero 
        if feat in feature_names
    ]
    
    logger.log_text(f"Erased features: {features_to_zero}")
    logger.log_text(f"Indices of zeroed columns: {indices_to_zero}")

    # 2. Extract, clone, and modify the data from the original dataloader
    all_modified_inputs = []
    all_labels = []

    for inputs, labels in test_dataloader:
        # Use .clone() to ensure we don't alter the original tensor in memory
        modified_inputs = inputs.clone()
        
        # 3. Zero out all rows (:) for the selected columns (indices_to_zero)
        if indices_to_zero: # Only perform the operation if there are indices to zero
            modified_inputs[:, indices_to_zero] = 0.0
            
        all_modified_inputs.append(modified_inputs)
        all_labels.append(labels)

    # Concatenate the batches back into single tensors
    X_erased_tensor = torch.cat(all_modified_inputs, dim=0)
    y_erased_tensor = torch.cat(all_labels, dim=0)

    # 4. Create the new modified Dataloader (the "copy")
    erased_dataset = TensorDataset(X_erased_tensor, y_erased_tensor)
    erased_dataloader = DataLoader(erased_dataset, batch_size=batch_size, shuffle=False)

    # 5. Run the metrics function with the NEW dataloader
    logger.log_text("Calculating metrics with the modified (erased) dataloader...")
    
    # Passed print_function as logger.log_text to keep everything in your log, 
    # but you can change it to the standard 'print' if you prefer.
    results = calculate_prediction_metrics(
        logger=logger, 
        model=model, 
        test_dataloader=erased_dataloader, 
        print_function=logger.log_text, 
        name=f"Feature_Erasure_Top_{top_n}_{name}", 
        is_multiclass=is_multiclass
    )
    
    return results
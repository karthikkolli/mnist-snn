use ndarray::{Array2, Axis, s};
use snn::ConvSNN;
use std::time::Instant;
use std::path::Path;
use tokio::fs;
use hf_hub::api::tokio::Api;
use std::fs::File;
use arrow::array::StructArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use image::ImageReader;
use std::io::Cursor;

async fn download_mnist_parquet() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let data_dir = Path::new("data");
    
    // Create data directory if it doesn't exist
    if !data_dir.exists() {
        fs::create_dir_all(data_dir).await?;
    }
    
    // Define parquet files
    let parquet_files = [
        ("mnist/train-00000-of-00001.parquet", "train-00000-of-00001.parquet"),
        ("mnist/test-00000-of-00001.parquet", "test-00000-of-00001.parquet"),
    ];
    
    // Initialize HF API
    let api = Api::new()?;
    let repo = api.dataset("ylecun/mnist".to_string());
    
    // Download parquet files if needed
    for (remote_path, local_name) in &parquet_files {
        let local_path = data_dir.join(local_name);
        
        if !local_path.exists() {
            println!("Downloading {}...", remote_path);
            let file_path = repo.get(remote_path).await?;
            // Copy from HF cache to our data directory
            tokio::fs::copy(&file_path, &local_path).await?;
            println!("✓ Downloaded {}", local_name);
        }
    }
    
    println!("MNIST dataset ready!");
    Ok(())
}

fn load_mnist_from_parquet(parquet_path: &Path) -> Result<(Vec<Vec<u8>>, Vec<u8>), Box<dyn std::error::Error>> {
    let file = File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;
    
    let mut images = Vec::new();
    let mut labels = Vec::new();
    
    // Read all record batches
    while let Some(batch_result) = reader.next() {
        let batch = batch_result?;
        
        // The schema has "image" (struct with "bytes" field) and "label" (int64) columns
        let image_column = batch.column(0).as_any().downcast_ref::<StructArray>()
            .ok_or("Failed to cast image column to StructArray")?;
        let label_column = batch.column(1).as_any().downcast_ref::<arrow::array::Int64Array>()
            .ok_or("Failed to cast label column to Int64Array")?;
        
        // Extract bytes from the struct array
        let bytes_array = image_column.column(0).as_any().downcast_ref::<arrow::array::BinaryArray>()
            .ok_or("Failed to cast bytes field to BinaryArray")?;
        
        // Process each row in the batch
        for i in 0..batch.num_rows() {
            let image_data = bytes_array.value(i);
            let label = label_column.value(i) as u8;
            
            // Decode PNG image
            let img = ImageReader::new(Cursor::new(image_data))
                .with_guessed_format()?
                .decode()?;
            
            // Convert to grayscale and get raw bytes
            let gray = img.to_luma8();
            let raw_pixels = gray.into_raw();
            
            images.push(raw_pixels);
            labels.push(label);
        }
    }
    
    Ok((images, labels))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Download MNIST parquet files if needed
    if let Err(e) = download_mnist_parquet().await {
        eprintln!("Failed to download MNIST dataset: {}", e);
        std::process::exit(1);
    }
    
    // Now run the synchronous training code
    run_training()?;
    Ok(())
}

fn run_training() -> Result<(), Box<dyn std::error::Error>> {
    // === Parse command line arguments ===
    let args: Vec<String> = std::env::args().collect();
    let epochs = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or_else(|_| {
            eprintln!("Invalid epochs argument. Usage: {} [epochs]", args[0]);
            eprintln!("Using default: 30 epochs");
            30
        })
    } else {
        30
    };
    
    println!("Running for {} epochs", epochs);
    
    // === Set random seed for reproducibility ===
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    let seed = 42; // Fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(seed);
    println!("Using random seed: {}", seed);
    
    // === Hyperparameters ===
    let mut learning_rate = 1e-2;
    let lr_decay = 0.95;
    let batch_size = 64;
    let steps = 10;
    
    // === Data Loading ===
    println!("Loading MNIST dataset from Parquet files...");
    
    let data_dir = Path::new("data");
    let train_parquet = data_dir.join("train-00000-of-00001.parquet");
    let test_parquet = data_dir.join("test-00000-of-00001.parquet");
    
    // Load training data
    let (train_images, train_labels) = load_mnist_from_parquet(&train_parquet)?;
    let train_size = train_images.len();
    
    // Load test data
    let (test_images, test_labels) = load_mnist_from_parquet(&test_parquet)?;
    let test_size = test_images.len();
    
    // Convert to ndarray format
    let trn_img = Array2::from_shape_vec(
        (train_size, 784),
        train_images.into_iter().flatten().collect()
    )?.mapv(|x| x as f64 / 255.0);
    
    let tst_img = Array2::from_shape_vec(
        (test_size, 784),
        test_images.into_iter().flatten().collect()
    )?.mapv(|x| x as f64 / 255.0);
    
    // Convert labels to one-hot encoding
    let mut trn_lbl = Array2::zeros((train_size, 10));
    for (i, &label) in train_labels.iter().enumerate() {
        trn_lbl[[i, label as usize]] = 1.0;
    }
    
    let mut tst_lbl = Array2::zeros((test_size, 10));
    for (i, &label) in test_labels.iter().enumerate() {
        tst_lbl[[i, label as usize]] = 1.0;
    }

    println!("Training set size: {}", train_size);
    println!("Test set size: {}", test_size);

    // === Network Initialization ===
    let mut network = ConvSNN::new_with_seed(seed);
    
    println!("\n=== CONVOLUTIONAL SNN ===");
    println!("Architecture:");
    println!("  Conv: 1x28x28 → 16x12x12 (5x5 kernels, stride=2)");
    println!("  LIF:  16x12x12 = 2304 spiking neurons");
    println!("  FC1:  2304 → 1024");
    println!("  LIF:  1024 spiking neurons");
    println!("  FC2:  1024 → 10");
    println!("==========================\n");

    // === Training Loop ===
    let mut best_test_acc = 0.0;
    let mut best_epoch = 0;
    let total_start = Instant::now();
    
    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;
        let mut correct_train = 0;
        let mut total_train = 0;
        
        // Shuffle training data
        let mut indices: Vec<usize> = (0..train_size).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);
        
        // Mini-batch training
        let num_batches = (train_size + batch_size - 1) / batch_size;
        
        for (batch_idx, batch_start) in (0..train_size).step_by(batch_size).enumerate() {
            let batch_end = (batch_start + batch_size).min(train_size);
            let batch_indices = &indices[batch_start..batch_end];
            
            // Prepare batch
            let mut batch_img = Array2::zeros((batch_indices.len(), 784));
            let mut batch_lbl = Array2::zeros((batch_indices.len(), 10));
            
            for (i, &idx) in batch_indices.iter().enumerate() {
                batch_img.slice_mut(s![i, ..]).assign(&trn_img.slice(s![idx, ..]));
                batch_lbl.slice_mut(s![i, ..]).assign(&trn_lbl.slice(s![idx, ..]));
            }
            
            // Train step
            let loss = network.train_step(&batch_img, &batch_lbl, learning_rate, steps);
            total_loss += loss * batch_indices.len() as f64;
            
            // Calculate training accuracy
            let outputs = network.forward(&batch_img, steps, false);
            let predictions = outputs.map_axis(Axis(1), |row| {
                row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
            });
            let true_labels = batch_lbl.map_axis(Axis(1), |row| {
                row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
            });
            
            correct_train += predictions.iter().zip(true_labels.iter())
                .filter(|(p, t)| p == t)
                .count();
            total_train += batch_indices.len();
            
            // Progress update
            if batch_idx % 100 == 0 && batch_idx > 0 {
                print!("\rEpoch {}/{} [{:3.0}%]", 
                       epoch + 1, epochs, 
                       batch_idx as f64 / num_batches as f64 * 100.0);
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
        }
        
        // === Evaluation ===
        let mut correct_test = 0;
        
        // Test in larger batches for efficiency
        for batch_start in (0..test_size).step_by(256) {
            let batch_end = (batch_start + 256).min(test_size);
            let batch_img = tst_img.slice(s![batch_start..batch_end, ..]);
            let batch_lbl = tst_lbl.slice(s![batch_start..batch_end, ..]);
            
            let outputs = network.forward(&batch_img.to_owned(), steps, false);
            
            let predictions = outputs.map_axis(Axis(1), |row| {
                row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
            });
            let true_labels = batch_lbl.map_axis(Axis(1), |row| {
                row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
            });
            
            correct_test += predictions.iter().zip(true_labels.iter())
                .filter(|(p, t)| p == t)
                .count();
        }
        
        // Calculate accuracies
        let train_acc = correct_train as f64 / total_train as f64 * 100.0;
        let test_acc = correct_test as f64 / test_size as f64 * 100.0;
        let avg_loss = total_loss / train_size as f64;
        let epoch_time = epoch_start.elapsed();
        
        println!("\rEpoch {}/{} - Loss: {:.4} - Train: {:.2}% - Test: {:.2}% - LR: {:.6} - Time: {:.1}s", 
                 epoch + 1, epochs, avg_loss, train_acc, test_acc, learning_rate, epoch_time.as_secs_f64());
        
        // Track best performance
        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            best_epoch = epoch + 1;
            println!("🎯 New best test accuracy: {:.2}%", best_test_acc);
        }
        
        // Learning rate decay
        if epoch > 0 && epoch % 5 == 0 {
            learning_rate *= lr_decay;
            println!("📉 Learning rate decayed to: {:.6}", learning_rate);
        }
        
        // Early stopping
        if test_acc >= 99.0 {
            println!("\n🏆 Achieved {:.2}% test accuracy! Training complete.", test_acc);
            break;
        }
    }
    
    let total_time = total_start.elapsed();
    
    println!("\n=== Training Complete ===");
    println!("Best test accuracy: {:.2}% (Epoch {})", best_test_acc, best_epoch);
    println!("Total training time: {:.1}s", total_time.as_secs_f64());
    println!("Average time per epoch: {:.1}s", total_time.as_secs_f64() / best_epoch as f64);
    
    Ok(())
}
use ndarray::{Array1, Array2, Array4, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use rayon::prelude::*;

/// Convolutional Spiking Neural Network
pub struct ConvSNN {
    // Network weights
    pub conv_weights: Array4<f64>,
    pub conv_bias: Array1<f64>,
    pub linear1_weights: Array2<f64>,
    pub linear1_bias: Array1<f64>,
    pub linear2_weights: Array2<f64>,
    pub linear2_bias: Array1<f64>,
    
    // LIF parameters
    pub lif1_decay: f64,
    pub lif2_decay: f64,
    pub k_threshold: f64,
    
    // Pre-computed convolution parameters
    conv_output_size: usize,
    conv_params: ConvParams,
    
    // Configuration
    pub num_threads: usize,
    
    // Spike histories
    conv_spike_history: Vec<Array2<f64>>,
    hidden_spike_history: Vec<Array2<f64>>,
}

struct ConvParams {
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    output_h: usize,
    output_w: usize,
}

impl ConvSNN {
    pub fn new() -> Self {
        Self::new_with_seed(42) // Default seed
    }
    
    pub fn new_with_seed(seed: u64) -> Self {
        use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        
        let num_threads = rayon::current_num_threads();
        
        // Set optimal thread pool size (often fewer threads = better)
        let optimal_threads = (num_threads / 2).max(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(optimal_threads)
            .build_global()
            .ok();
        
        // Architecture
        let out_channels = 16;
        let kernel_size = 5;
        let stride = 2;
        let output_h = 12;
        let output_w = 12;
        let conv_output_size = out_channels * output_h * output_w;
        
        // Initialize weights
        let fan_in = kernel_size * kernel_size;
        let std_dev = (2.0 / fan_in as f64).sqrt();
        let conv_weights = Array4::random_using(
            (out_channels, 1, kernel_size, kernel_size),
            Normal::new(0.0, std_dev).unwrap(),
            &mut rng
        );
        let conv_bias = Array1::zeros(out_channels);
        
        // Initialize linear layers
        let scale1 = (2.0 / (conv_output_size + 1024) as f64).sqrt();
        let linear1_weights = Array2::random_using((conv_output_size, 1024), Normal::new(0.0, scale1).unwrap(), &mut rng);
        let linear1_bias = Array1::zeros(1024);
        
        let scale2 = (2.0 / (1024 + 10) as f64).sqrt();
        let linear2_weights = Array2::random_using((1024, 10), Normal::new(0.0, scale2).unwrap(), &mut rng);
        let linear2_bias = Array1::zeros(10);
        
        Self {
            conv_weights,
            conv_bias,
            linear1_weights,
            linear1_bias,
            linear2_weights,
            linear2_bias,
            lif1_decay: 0.9,
            lif2_decay: 0.9,
            k_threshold: 0.8,
            conv_output_size,
            conv_params: ConvParams {
                out_channels,
                kernel_size,
                stride,
                output_h,
                output_w,
            },
            num_threads: optimal_threads,
            conv_spike_history: Vec::new(),
            hidden_spike_history: Vec::new(),
        }
    }
    
    /// Convolution forward pass
    fn conv_forward(&self, input_batch: &Array2<f64>) -> Array2<f64> {
        let batch_size = input_batch.shape()[0];
        let ConvParams { out_channels, kernel_size, stride, output_h, output_w } = &self.conv_params;
        
        // Process batch samples
        let outputs: Vec<Vec<f64>> = (0..batch_size).into_par_iter()
            .map(|b| {
                let input = input_batch.row(b);
                let input_2d = input.into_shape((28, 28)).unwrap();
                let mut output = vec![0.0; self.conv_output_size];
                
                // Convolution for this sample
                for oc in 0..*out_channels {
                    let kernel = self.conv_weights.slice(s![oc, 0, .., ..]);
                    let bias = self.conv_bias[oc];
                    
                    for oh in 0..*output_h {
                        for ow in 0..*output_w {
                            let h_start = oh * stride;
                            let w_start = ow * stride;
                            
                            let mut sum = 0.0;
                            for kh in 0..*kernel_size {
                                for kw in 0..*kernel_size {
                                    let h = h_start + kh;
                                    let w = w_start + kw;
                                    if h < 28 && w < 28 {
                                        sum += input_2d[[h, w]] * kernel[[kh, kw]];
                                    }
                                }
                            }
                            
                            let out_idx = oc * output_h * output_w + oh * output_w + ow;
                            output[out_idx] = sum + bias;
                        }
                    }
                }
                
                output
            })
            .collect();
        
        // Convert to Array2
        let flat: Vec<f64> = outputs.into_iter().flatten().collect();
        Array2::from_shape_vec((batch_size, self.conv_output_size), flat).unwrap()
    }
    
    /// Forward pass
    pub fn forward(&mut self, inputs: &Array2<f64>, steps: usize, is_training: bool) -> Array2<f64> {
        let batch_size = inputs.shape()[0];
        
        // Pre-allocate arrays
        let mut conv_potentials = Array2::zeros((batch_size, self.conv_output_size));
        let mut hidden_potentials = Array2::zeros((batch_size, 1024));
        let mut conv_spike_counts = Array2::zeros((batch_size, self.conv_output_size));
        let mut hidden_spike_counts = Array2::zeros((batch_size, 1024));
        
        if is_training {
            self.conv_spike_history.clear();
            self.hidden_spike_history.clear();
        }
        
        // Pre-generate all random numbers for efficiency with a fixed seed
        let random_vals: Vec<Vec<f64>> = if steps > 5 {
            use ndarray_rand::rand::{SeedableRng, rngs::StdRng, Rng};
            let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
            (0..steps-5).map(|_| {
                (0..inputs.len()).map(|_| rng.gen_range(0.0..1.0)).collect()
            }).collect()
        } else {
            vec![]
        };
        
        for t in 0..steps {
            // Generate spikes
            let spike_prob = if t < 5 {
                inputs.mapv(|p| if p > (1.0 - t as f64 / 5.0) { 1.0 } else { 0.0 })
            } else {
                let mut spikes = Array2::zeros(inputs.dim());
                let rand_idx = t - 5;
                if rand_idx < random_vals.len() {
                    for (spike, (&input, &rand)) in spikes.iter_mut()
                        .zip(inputs.iter().zip(random_vals[rand_idx].iter())) {
                        *spike = if rand < input { 1.0 } else { 0.0 };
                    }
                }
                spikes
            };
            
            // Convolution
            let conv_input = self.conv_forward(&spike_prob);
            
            // LIF dynamics
            conv_potentials *= self.lif1_decay;
            conv_potentials += &conv_input;
            
            let conv_spikes = conv_potentials.mapv(|v| if v > self.k_threshold { 1.0 } else { 0.0 });
            conv_spike_counts += &conv_spikes;
            
            // Reset
            for (pot, spike) in conv_potentials.iter_mut().zip(conv_spikes.iter()) {
                if *spike > 0.0 { *pot = 0.0; }
            }
            
            // Hidden layer (every other timestep)
            if t % 2 == 0 {
                let hidden_input = conv_spike_counts.dot(&self.linear1_weights) + &self.linear1_bias;
                
                hidden_potentials *= self.lif2_decay;
                hidden_potentials += &(&hidden_input * 0.1);
                
                let hidden_spikes = hidden_potentials.mapv(|v| if v > self.k_threshold { 1.0 } else { 0.0 });
                hidden_spike_counts += &hidden_spikes;
                
                // Soft reset
                for (pot, spike) in hidden_potentials.iter_mut().zip(hidden_spikes.iter()) {
                    if *spike > 0.0 { *pot *= 0.5; }
                }
            }
        }
        
        // Normalize
        conv_spike_counts /= steps as f64;
        hidden_spike_counts /= steps as f64;
        
        if is_training {
            self.conv_spike_history.push(conv_spike_counts.clone());
            self.hidden_spike_history.push(hidden_spike_counts.clone());
        }
        
        // Output computation
        let output = hidden_spike_counts.dot(&self.linear2_weights) + &self.linear2_bias;
        
        // Softmax
        let max_vals = output.map_axis(Axis(1), |row| 
            row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );
        let exp_vals = (&output - &max_vals.insert_axis(Axis(1))).mapv(f64::exp);
        let sum_exp = exp_vals.sum_axis(Axis(1));
        
        &exp_vals / &sum_exp.insert_axis(Axis(1))
    }
    
    /// Training step
    pub fn train_step(
        &mut self,
        inputs: &Array2<f64>,
        labels: &Array2<f64>,
        learning_rate: f64,
        steps: usize,
    ) -> f64 {
        // Forward pass
        let outputs = self.forward(inputs, steps, true);
        
        // Loss computation (simple reduction, no need for parallel)
        let batch_size = inputs.shape()[0];
        let mut loss = 0.0;
        for b in 0..batch_size {
            for i in 0..10 {
                if labels[[b, i]] > 0.0 {
                    loss -= (outputs[[b, i]] + 1e-10).ln();
                }
            }
        }
        loss /= batch_size as f64;
        
        // Gradient computation and weight updates
        let output_error = &outputs - labels;
        
        if let (Some(hidden_spikes), Some(conv_spikes)) = 
            (self.hidden_spike_history.last(), self.conv_spike_history.last()) {
            
            // All matrix operations here use optimized BLAS
            let grad2 = hidden_spikes.t().dot(&output_error);
            let bias_grad2 = output_error.sum_axis(Axis(0));
            
            let hidden_error = output_error.dot(&self.linear2_weights.t());
            let grad1 = conv_spikes.t().dot(&hidden_error);
            
            // Weight updates
            self.linear2_weights.scaled_add(-learning_rate, &grad2);
            self.linear2_bias.scaled_add(-learning_rate, &bias_grad2);
            self.linear2_weights *= 0.9999;
            
            self.linear1_weights.scaled_add(-learning_rate * 0.1, &grad1);
            self.linear1_weights *= 0.9999;
        }
        
        loss
    }
}
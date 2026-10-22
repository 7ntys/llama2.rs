pub fn rmsnorm(x: &[f32], o: &mut [f32], weights: &[f32], size: usize) {
    let mut sum_square: f32 = 0.0;

    for i in 0..size {
        sum_square += x[i] * x[i];
    }

    let avg_square = sum_square / (size as f32) + 1e-5f32; // Prevent division by 0
    let factor = 1.0 / avg_square.sqrt();

    for i in 0..size {
        o[i] = weights[i] * (x[i] * factor);
    }
}

pub fn softmax(x: &mut [f32], size: usize) {
    let mut max_val = x[0];

    // Get max of buffer
    for i in 0..size {
        if x[i] > max_val {
            max_val = x[i];
        }
    }

    // First step: exp
    let mut sum = 0.0;
    for i in 0..size {
        x[i] = f32::exp(x[i] - max_val);
        sum += x[i];
    }

    // Normalize
    for i in 0..size {
        x[i] /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_norm() {
        let x: Vec<f32> = vec![3.0, 4.0];
        let mut o: Vec<f32> = vec![0.0; 2];
        let w: Vec<f32> = vec![1.0, 2.0];

        rmsnorm(&x, &mut o, &w, 2);

        assert!((o[0] - 0.849).abs() < 1e-3);
        assert!((o[1] - 2.263).abs() < 1e-3);
    }

    #[test]
    fn verify_softmax_small_number() {
        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0];
        softmax(&mut x, 3);

        let expected = vec![0.0900, 0.2447, 0.6652];

        for i in 0..3 {
            assert!((x[i] - expected[i]).abs() < 1e-3)
        }
    }

    #[test]
    fn verify_softmax_big_number() {
        let mut x: Vec<f32> = vec![1000.0, 1001.0, 1002.0];
        softmax(&mut x, 3);

        let expected = vec![0.0900, 0.2447, 0.6652];

        for i in 0..3 {
            assert!((x[i] - expected[i]).abs() < 1e-3)
        }
    }
}

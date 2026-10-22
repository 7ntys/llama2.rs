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
}

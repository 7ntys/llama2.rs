pub fn rmsnorm(x: &[f32], w: &[f32], o: &mut [f32], size: usize) {
    let mut sum_square: f32 = 0.0;

    for i in 0..size {
        sum_square += x[i] * x[i];
    }

    let avg_square = sum_square / (size as f32) + 1e-5f32; // Prevent division by 0
    let factor = 1.0 / avg_square.sqrt();

    for i in 0..size {
        o[i] = w[i] * (x[i] * factor);
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

pub fn matmul(x: &[f32], w: &[f32], o: &mut [f32], n: usize, d: usize) {
    for i in 0..d {
        let mut val = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        o[i] = val;
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut score = 0.0;
    for i in 0..a.len() {
        score += a[i] * b[i];
    }
    return score;
}

pub fn silu(x: f32) -> f32 {
    let divisor = 1.0 + f32::exp(-x);
    return x * (1.0 / divisor);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_norm() {
        let x: Vec<f32> = vec![3.0, 4.0];
        let mut o: Vec<f32> = vec![0.0; 2];
        let w: Vec<f32> = vec![1.0, 2.0];

        rmsnorm(&x, &w, &mut o, 2);

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

    #[test]
    fn verify_matmul() {
        let x: Vec<f32> = vec![1.0, 0.0, 2.0];
        let w: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = 2;
        let n = 3;

        let mut o: Vec<f32> = vec![0.0; d];
        matmul(&x, &w, &mut o, n, d);

        let expected = vec![7.0, 16.0];

        for i in 0..d {
            assert_eq!(o[i], expected[i])
        }
    }

    #[test]
    fn verify_silu() {
        assert!((silu(2.0) - 1.762).abs() <= 1e-3)
    }
}

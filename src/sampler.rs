pub fn argmax(logits: &[f32]) -> usize {
    // Go through all logits and get the max one:
    let mut idx = 0;
    for i in 0..logits.len() {
        if logits[i] > logits[idx] {
            idx = i;
        }
    }
    return idx;
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn verify_arg_max() {
        let logits: Vec<f32> = vec![2.0, 5.0, 5.0, -1.0];
        assert_eq!(argmax(&logits), 1);
    }
}

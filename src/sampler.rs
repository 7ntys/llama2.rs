use crate::kernels::softmax;

pub struct Sampler {
    pub(crate) temperature: f32,
    pub(crate) rng_state: u64,
    pub(crate) top_p: f32,
}

impl Sampler {
    pub fn build_sampler(temperature: f32, seed: u64, top_p: f32) -> Self {
        Self {
            temperature: temperature,
            rng_state: seed,
            top_p,
        }
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature == 0.0 {
            return self.argmax(logits);
        }

        // Divide all logits by temperature:
        for i in 0..logits.len() {
            logits[i] /= self.temperature;
        }

        softmax(logits, logits.len());
        let mut target = self.get_random_f32();
        let mut curr = 0.0;

        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut candidates = self.get_top_p_optimisation(logits);
            candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

            // get the top-p candidates:
            let mut cumul_prob = 0.0;
            let mut i = 0;

            while cumul_prob <= self.top_p && i < candidates.len() {
                cumul_prob += candidates[i].1;
                i += 1;
            }

            let kept = &candidates[0..i];
            target *= cumul_prob;
            for (i, prob) in kept {
                curr += prob;

                if curr > target {
                    return *i;
                }
            }

            return kept[kept.len() - 1].0;
        } else {
            for i in 0..logits.len() {
                curr += logits[i];

                if curr > target {
                    return i;
                }
            }

            return logits.len() - 1;
        }
    }

    fn get_top_p_optimisation(&mut self, logits: &[f32]) -> Vec<(usize, f32)> {
        let mut out: Vec<(usize, f32)> = vec![];

        let n = logits.len();
        let target = (1.0 - self.top_p) / (n - 1) as f32;
        for i in 0..n {
            // Check if worth sampling :
            if logits[i] >= target {
                out.push((i, logits[i]));
            }
        }

        return out;
    }

    fn get_random_f32(&mut self) -> f32 {
        let mut random = self.get_random_u32();
        random = random >> 8;
        let random_f32 = random as f32;
        return random_f32 / 16777216.0;
    }

    // Copy 'run.c' random generator for benchmarking purposes
    fn get_random_u32(&mut self) -> u32 {
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        return (self.rng_state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32;
    }

    fn argmax(&self, logits: &[f32]) -> usize {
        // Go through all logits and get the max one:
        let mut idx = 0;
        for i in 0..logits.len() {
            if logits[i] > logits[idx] {
                idx = i;
            }
        }
        return idx;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn verify_arg_max() {
        let mut logits: Vec<f32> = vec![2.0, 5.0, 5.0, -1.0];
        let mut sampler = Sampler::build_sampler(0.0, 42, 0.9);
        assert_eq!(sampler.sample(&mut logits), 1);
    }
}

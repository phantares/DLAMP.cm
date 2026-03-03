import numpy as np
import onnxruntime as ort
import yaml


class EDMSampler:
    def __init__(self, model_path, config_path, sigma_min=0.002, sigma_max=80):
        self.session = ort.InferenceSession(model_path)

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

            self.sigma_max = min(float(cfg["sigma_max"]["value"]), sigma_max)
            self.sigma_min = max(float(cfg["sigma_min"]["value"]), sigma_min)

    def sample(
        self,
        noise,
        conditions,
        num_steps=18,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):

        required = {"single", "upper", "time", "column_top", "column_left"}
        if not required.issubset(conditions.keys()):
            raise ValueError(f"Missing: {required - conditions.keys()}")

        step_indices = np.arange(num_steps)
        t_steps = (
            self.sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))
        ) ** rho
        t_steps = np.append(t_steps, 0.0)  # t_N = 0

        x_next = noise * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + np.sqrt(
                t_hat**2 - t_cur**2
            ) * S_noise * np.random.randn(*x_cur.shape).astype(noise.dtype)

            denoised = self._run_model(x_hat, t_hat, conditions)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < num_steps - 1:
                denoised = self._run_model(x_next, t_next, conditions)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    def _run_model(self, noise, sigma, conditions):
        inputs = conditions
        inputs["noise"] = noise.astype(np.float32)
        inputs["sigma"] = np.full(
            (noise.shape[0], noise.shape[1], 1, 1, 1, 1), sigma, dtype=np.float32
        )

        return self.session.run(None, inputs)[0]

import jax
import jax.numpy as jnp

class Hebbian:
    def __init__(self, states, corrupt_states):
        self.states = states
        self.corrupt_states = corrupt_states
        self.dim = len(states[0].flatten())
        self.num_states = len(states)
        self.key = jax.random.PRNGKey(40)

        self.mod_state_arr = states.reshape(self.num_states, self.dim)
        self.W_unmod_arr = 1/self.dim * (self.mod_state_arr.T @ self.mod_state_arr)
        self.W_arr = self.W_unmod_arr.at[jnp.diag_indices(self.dim)].set(0)
        

    def update(self):
        W_arr = self.W_arr
        s_arrs = self.corrupt_states
        
        @jax.jit
        def step_fn(state, key):
            state_flat = state.flatten()
            h_arr_0 = jnp.dot(W_arr, state_flat)
            h_arr_sign = jnp.sign(h_arr_0)
            h_arr_end = jnp.where(h_arr_0 == 0, state_flat, h_arr_sign)

            probability = 0.5
            mask = jax.random.bernoulli(key, p=probability, shape=h_arr_end.shape)
            new_state_flat = jnp.where(mask, h_arr_end, state_flat)
            new_state = new_state_flat.reshape(state.shape)

            return new_state, new_state

        num_steps = 15
        final_states = []
        histories = []

        for s_arr in s_arrs:
            key_seq, self.key = jax.random.split(self.key)
            scan_keys = jax.random.split(key_seq, num_steps)
            
            fin_st, his = jax.lax.scan(step_fn, s_arr, scan_keys)
            final_states.append(fin_st)
            histories.append(his)

        return final_states, histories


def state_corruption(state):
    corruption_key = jax.random.PRNGKey(16)
    corruption_probability = 0.45
    mask = jax.random.bernoulli(corruption_key, corruption_probability, state.shape)

    return jnp.where(mask, state * -1, state)


state_masks = jax.random.bernoulli(jax.random.PRNGKey(4), 0.5, shape=(5, 100, 100))
states = jnp.where(state_masks, 1, -1)
state_1, state_2, state_3, state_4, state_5 = states
cor_states = [state_corruption(state_2), state_corruption(state_5)]


for i, (cor_st, original_st) in enumerate(zip(cor_states, [state_2, state_5])):
    similarity = jnp.mean(cor_st == original_st)
    print(f"State {i} initial similarity: {similarity:.2%}")

heb = Hebbian(states, cor_states)
final_states, histories = heb.update()

for i, (final_st, original_st) in enumerate(zip(final_states, [state_2, state_5])):
    similarity = jnp.mean(final_st == original_st)
    print(f"State {i} similarity: {similarity:.2%}")

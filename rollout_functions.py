import numpy as np
import copy

def condensed_rollout(env,
                      agent,
                      max_path_length=np.inf,
                      render=False,
                      render_kwargs=None,
                      preprocess_func=None,
                      get_action_kwargs=None,
                      postprocess_func=None,
                      reset_callback=None):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_func is None:
        preprocess_func = lambda x: x
        
    data = []
    path_length = 0
    agent.reset()
    
    o = env.reset()
    rewards = []
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    
    while path_length < max_path_length:
        o_for_agent = preprocess_func(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)
        
        # condense state and action into single observation
        o['a'] = a
        
        if postprocess_func:
            postprocess_func(env, agent, o)
            
        next_o, r, done, env_info = env.step(copy.deepcopy(a))
        o['next_s'] = next_o.x
        o['r'] = r
        o['t'] = done
        
        rewards.append(r.sum().item())
        if render:
            env.render(**render_kwargs)
        
        data.append(o)
        path_length += 1
        if done:
            break
        o = next_o        
        
    return {'observations': data, 'rewards': rewards, 'size': len(data)}

def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    dones = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    
    o = env.reset()
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)
        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)
            
        next_o, r, done, env_info = env.step(copy.deepcopy(a))
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminal = False
        if done:
            # terminal=False if TimeLimit caused termination
            if not env_info.pop('TimeLimit.truncated', False):
                terminal = True
        terminals.append(terminal)
        dones.append(done)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if done:
            break
        o = next_o
        
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
        
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        dones=dones,
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )

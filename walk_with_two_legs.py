import gymnasium as gym
from stable_baselines3 import PPO

# --- 1. 학습 (빠른 모드) ---
print("--- 1. 학습 시작 (Walker2d) ---")
# 1. 환경 이름을 "Walker2d-v4"로 수정!
env_train = gym.make("Walker2d-v4") 
model = PPO("MlpPolicy", env_train, verbose=1, device="cpu")

# Walker는 더 복잡하므로 100,000 스텝은 '맛보기'일 수 있습니다.
model.learn(total_timesteps=1000000) 

# 2. 모델 파일 이름 수정
model.save("ppo_walker_model")
env_train.close()
print("--- 1. 학습 완료 및 모델 저장 ---")


# --- 2. 시청 (학습된 결과 확인) ---
print("--- 2. 저장된 모델 불러와서 시청 ---")
# 2. 불러올 모델 이름 수정
model = PPO.load("ppo_walker_model")

# 1. 시청 환경 이름도 "Walker2d-v4"로 수정!
env_watch = gym.make("Walker2d-v4", render_mode="human")
obs, info = env_watch.reset()

for _ in range(2000): # 2000 스텝 동안 지켜보기
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_watch.step(action)
    
    if terminated or truncated:
        print("에피소드 종료. 리셋합니다.")
        obs, info = env_watch.reset()

env_watch.close()
print("--- 2. 시청 종료 ---")
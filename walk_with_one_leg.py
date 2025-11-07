import gymnasium as gym
from stable_baselines3 import PPO

# --- 1. 학습 (빠른 모드) ---
print("--- 1. 학습 시작 (렌더링 없음) ---")
# render_mode를 None (기본값)으로 설정하거나 아예 삭제
env_train = gym.make("Hopper-v5") 
model = PPO("MlpPolicy", env_train, verbose=1, device="cpu") # 경고 메시지대로 CPU로 변경

# 시간을 더 늘려서 성능을 높여봅시다 (예: 100,000)
model.learn(total_timesteps=100000) 

# 학습된 모델을 파일로 저장
model.save("ppo_hopper_model")
env_train.close()
print("--- 1. 학습 완료 및 모델 저장 ---")


# --- 2. 시청 (학습된 결과 확인) ---
print("--- 2. 저장된 모델 불러와서 시청 ---")
# 저장된 모델 불러오기
model = PPO.load("ppo_hopper_model")

# 이번에는 render_mode="human"으로 시청용 환경을 생성
env_watch = gym.make("Hopper-v5", render_mode="human")
obs, info = env_watch.reset()

for _ in range(2000): # 2000 스텝 동안 지켜보기
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_watch.step(action)
    
    if terminated or truncated:
        print("에피소드 종료. 리셋합니다.")
        obs, info = env_watch.reset()

env_watch.close()
print("--- 2. 시청 종료 ---")
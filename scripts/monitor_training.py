"""
실시간 학습 진행 상황 모니터링
"""
import pandas as pd
import time
import os
from pathlib import Path

def monitor_training(log_file='large_scale_training_log.csv', refresh_interval=5):
    """학습 진행 상황 실시간 모니터링"""
    
    print("\n" + "="*80)
    print("📊 학습 진행 상황 모니터링")
    print("="*80 + "\n")
    print("Ctrl+C를 눌러 종료하세요.\n")
    
    last_epoch = 0
    
    try:
        while True:
            if os.path.exists(log_file):
                # 로그 읽기
                df = pd.read_csv(log_file)
                
                if len(df) > 0:
                    latest = df.iloc[-1]
                    current_epoch = len(df)
                    
                    # 새로운 epoch 완료 시에만 출력
                    if current_epoch > last_epoch:
                        last_epoch = current_epoch
                        
                        # 진행률
                        total_epochs = 30  # 설정값
                        progress = current_epoch / total_epochs * 100
                        
                        # 진행 바
                        bar_length = 50
                        filled = int(bar_length * current_epoch / total_epochs)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        
                        print(f"\r{'='*80}")
                        print(f"Epoch {current_epoch}/{total_epochs} ({progress:.1f}%)")
                        print(f"[{bar}]")
                        print(f"{'='*80}")
                        print(f"Train Loss: {latest['train_loss']:.4f} | Train Acc: {latest['train_acc']:.2f}%")
                        print(f"Val Loss:   {latest['val_loss']:.4f} | Val Acc:   {latest['val_acc']:.2f}%")
                        
                        # Best 기록
                        best_val_acc = df['val_acc'].max()
                        print(f"\n🏆 Best Val Acc: {best_val_acc:.2f}%")
                        print("="*80 + "\n")
                else:
                    print(f"\r⏳ 학습 시작 대기 중...", end='', flush=True)
            else:
                print(f"\r⏳ 로그 파일 생성 대기 중...", end='', flush=True)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ 모니터링 종료")
        
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            if len(df) > 0:
                print("\n📊 최종 상태:")
                print(df.tail())

if __name__ == "__main__":
    monitor_training()

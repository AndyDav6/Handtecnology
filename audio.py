import pygame
import os
import threading

def play_sound(sound_path = './soundo/Wile.mp3'):
    if not os.path.exists(sound_path):
        print(f"[ERROR] No se ecnontro el archivo de sonido: {sound_path}")
        return
    
    def _play():
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pass
        
        except Exception as e:
            print(f"[ERROR] Al reproducir el sonido: {e}")

    thread = threading.Thread(target = _play)
    thread.start()

    
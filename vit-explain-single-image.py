# Questo script avvia vit_gcn_explain.py per generare spiegazioni su un'immagine specificata,
# creando un dataset temporaneo con l'immagine in una cartella nascosta che viene eliminata al termine

import os
import cv2
import shutil
import chardet
import argparse
from PIL import Image

def read_file_with_fallback(info_file):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'windows-1252']
    for encoding in encodings:
        try:
            with open(info_file, 'r', encoding=encoding) as file:
                return file.readlines()
        except UnicodeDecodeError:
            print(f"Errore con la codifica {encoding}, tentando con la successiva...")
    raise Exception("Impossibile decodificare il file con le codifiche tentate.")

def read_file_with_chardet(info_file):
    with open(info_file, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print(f"Codifica rilevata: {encoding}")
        
        with open(info_file, 'r', encoding=encoding) as file:
            return file.readlines()

def read_file(info_file):
    try:
        return read_file_with_fallback(info_file)
    except Exception as e:
        print(str(e))
        return read_file_with_chardet(info_file)

def verify_path_exists(path):
    # Verifica che il path esista
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Il path specificato non esiste: {abs_path}")
    return abs_path

def convert_to_jpg(image_path):
    # Converte l'immagine in formato JPG se necessario
    try:
        img = Image.open(image_path)
        if img.format == 'JPEG':
            return image_path
            
        directory = os.path.dirname(image_path)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        jpg_path = os.path.join(directory, f"{filename}.jpg")
        
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(jpg_path, 'JPEG', quality=95)
        print(f"Immagine convertita in JPG: {jpg_path}")
        
        return jpg_path
    except Exception as e:
        raise Exception(f"Errore durante la conversione in JPG: {str(e)}")

def create_hidden_folder_and_process_image(image_path):
    # Verifica che l'immagine esista
    image_path = verify_path_exists(image_path)
    
    # Converti l'immagine in JPG se necessario
    jpg_image_path = convert_to_jpg(image_path)
    
    # Creazione della cartella nascosta
    hidden_folder = os.path.join(os.path.dirname(jpg_image_path), f".{os.path.basename(jpg_image_path).split('.')[0]}")
    os.makedirs(hidden_folder, exist_ok=True)
    
    # Copia dell'immagine nella cartella nascosta
    img_filename = os.path.basename(jpg_image_path)
    hidden_image_path = os.path.join(hidden_folder, img_filename)
    shutil.copy(jpg_image_path, hidden_image_path)
    
    # Crea il file .txt con le dimensioni dell'immagine
    image = cv2.imread(hidden_image_path)
    if image is None:
        raise ValueError(f"Impossibile leggere l'immagine: {hidden_image_path}")
    
    height, width = image.shape[:2]
    txt_filename = os.path.splitext(img_filename)[0] + '.txt'
    txt_file_path = os.path.join(hidden_folder, txt_filename)
    with open(txt_file_path, 'w') as file:
        file.write(f"1 0 0 {height} {width}")
        
    # Se abbiamo creato una nuova immagine JPG e non è il file originale, la eliminiamo
    if jpg_image_path != image_path:
        os.remove(jpg_image_path)
        
    return os.path.abspath(hidden_folder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Percorso dell\'immagine')    
    args = parser.parse_args()
    
    try:
        # Crea la cartella nascosta e copia l'immagine
        hidden_folder = create_hidden_folder_and_process_image(args.image_path)
        
        if not os.path.exists(hidden_folder):
            raise FileNotFoundError(f"La cartella nascosta non è stata creata correttamente: {hidden_folder}")
            
        # Chiamata allo script vit-explain.py
        exit_code = os.system(f"python vit_gcn_explain.py --hidden 256 --classes 8 --input_path \"{hidden_folder}\" --no_cuda")
        
        if exit_code != 0:
            raise RuntimeError(f"Errore nell'esecuzione di vit-explain.py. Codice di uscita: {exit_code}")
            
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
        raise
        
    finally:
        # Rimuove la cartella nascosta dopo il processo
        if 'hidden_folder' in locals() and os.path.exists(hidden_folder):
            shutil.rmtree(hidden_folder)
            print(f"Cartella nascosta {hidden_folder} eliminata.")

if __name__ == "__main__":
    main()
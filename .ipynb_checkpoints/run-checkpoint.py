from fun import *

## ['run.py' <imagem> <tipo de filtro> <mode> <plano rgb> <imagem filtrada> <pt>

if __name__ == "__main__":
    if sys.argv[1][-4:]==".png":
        print("Iniciando...")
        func["filtrar"](sys.argv[1],
                        sys.argv[2],
                        sys.argv[3],
                        sys.argv[4],
                        sys.argv[5])
        
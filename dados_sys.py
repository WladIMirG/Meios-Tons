from imports import *


class Imagem:
    nome         : str = None
    nome_path    : str = None
    w            : int = None # x
    h            : int = None # y
    ch           : int = None
    array        : np.ndarray = None
    carray       : np.ndarray = None
    r            : tuple = None
    g            : tuple = None
    b            : tuple = None
    
    def imag_up(self, nome : str) -> None:
        self.nome = nome
        self.nome_path = nome_path = os.getcwd()+"/imagens/"+self.nome
        self.array = np.array(cv2.imread(nome_path))
        self.h, self.w, self.ch = self.shape()
    
    def weight(self) -> int:
        return os.path.getsize(self.nome_path)
    
    def shape(self) -> tuple:
        return self.array.shape
    
    def img_plot(self, array, pt = "cv2", title = "Original"):
        if pt == "cv2":
            cv2.imshow("Orginal", array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if pt == "plt":
            fig = plt.subplots(figsize=(10,10))
            # org = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            plt.imshow(array)
            plt.title(title)
            plt.show()
    
    def img_save(self, nome : str = None) -> None:
        if nome is not None:
            cv2.imwrite("imagens/"+nome, self.array)
        else:
            cv2.imwrite("imagens/"+"_"+self.nome, self.array)
    
    def img_convert(self, CS : Any = cv2.COLOR_BGR2YCrCb):
        self.carray = cv2.cvtColor(self.array, CS)
        # print (self.carray.shape)
    
    def set_normfor(self, nome : str, ext : str) -> None:
        self.nome = nome
        self.ext = ext
    

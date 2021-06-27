import cv2
import numpy as np
import os
from PIL import Image
import time

# Panduan bantuan Pengkoreksian :
# Method SVD
# A. Perlakuan SVD dapat dibagi menjadi 2 :
# 1. Bentuk Matriks Vertical
# 2. Bentuk Matriks Horizontal
# Yang membedakan pada saat proses transpose saja
# B .Method SVD perlu mencari U, sigma, dan V untuk mendekomposisi suatu matriks menjadi 3 matriks dekomposisi
# U dicari dengan fungsi getU, sigma di cari langsung dengan menggunakan linalg.eig, dan V dicari dengan fungsi getV
 
#Fungsi untuk mendapatkan matriks U pada matriks vertical, atau V pada matriks horizontal
def getU(matriks, v, sigma):
    u = []
    for i in range(len(v)):
        ui = (1/sigma[i])*np.dot(matriks, v[:,i])
        u.append(ui)
        
    return np.transpose(np.array(u))

#Fungsi untuk mendapatkan matriks U pada matriks horizontal, atau V pada matriks vertical
def getV(matriks, u, sigma):
    v = []
    for i in range(len(u)):
        vi = (1/sigma[i])*np.dot(matriks.transpose(), u[:,i])
        v.append(vi)

    return np.transpose(np.array(v))

#Fungsi untuk compressing image 
def creatingImgCompress(m,n,sigma, u, v, k):
    #Buat matriks null terlebih dahulu, bisa menggunakan bantuan rank dan null tetapi sama saja dengan menggunakan panjang baris dan kolom
    B = np.zeros((m,n))
    matriksCompress = np.zeros((m, n), dtype=np.uint8)

    for i in range(k):
        B = np.add(B, sigma[i]*np.outer(u[:, i], v[:, i]))

    # Suatu image memang dapat di komposisi menjadi angka integer antara 0..255
    for i in range(len(B)):
        for j in range(len(B[0])):
            if B[i][j] > 255:
                matriksCompress[i][j] = 255
            elif B[i][j] < 0:
                matriksCompress[i][j] = 0
            else:
                matriksCompress[i][j] = B[i][j]
  
    return matriksCompress

#fungsi draw untuk menampilkan GUI
def draw(gambar):
    if (gambar == 1):
        print("  _                                                                                                   _  ")
        print(" | |                                                                                                 | | ")
        print(" | |______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______| | ")
        print(" | |______|______|______|______|______|______|______|______|______|______|______|______|______|______| | ")
        print(" | |                                                                                                 | | ")
        print(" | |                                                                                                 | | ")
        print(" | |                                                                                                 | | ")
        print(" |_|  _____                                 _____                                                    |_| ")
        print(" | | |_   _|                               / ____|                                                   | | ")
        print(" | |   | |  _ __ ___   __ _  __ _  ___    | |     ___  _ __ ___  _ __  _ __ ___  ___ ___  ___  _ __  | | ")
        print(" | |   | | | '_ ` _ \ / _` |/ _` |/ _ \   | |    / _ \| '_ ` _ \| '_ \| '__/ _ \/ __/ __|/ _ \| '__| | | ")
        print(" | |  _| |_| | | | | | (_| | (_| |  __/   | |___| (_) | | | | | | |_) | | |  __/\__ \__ \ (_) | |    | | ")
        print(" | | |_____|_| |_| |_|\__,_|\__, |\___|    \_____\___/|_| |_| |_| .__/|_|  \___||___/___/\___/|_|    | | ")
        print(" | |                         __/ |                              | |                                  | | ")
        print(" |_|                        |___/                               |_|                                  |_| ")
        print(" | |                                                                                                 | | ")
        print(" | |______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______| | ")
        print(" | |______|______|______|______|______|______|______|______|______|______|______|______|______|______| | ")
        print(" | |                                                                                                 | | ")
        print(" | |                                                                                                 | | ")
        print(" | |                                                                                                 | | ")
        print(" |_|                                                                                                 |_| ")
    else:
        print("______           _     __                                                                 __     _____                 _            ")
        print("| ___ \         | |   / /                                                                 \ \   |  __ \               | |           ")
        print("| |_/ / __ _  __| |  / /_____ ______ ______ ______ ______ ______ ______ ______ ______ _____\ \  | |  \/ ___   ___   __| |           ")
        print("| ___ \/ _` |/ _` | < <______|______|______|______|______|______|______|______|______|______> > | | __ / _ \ / _ \ / _` |           ")
        print("| |_/ / (_| | (_| |  \ \                                                                   / /  | |_\ \ (_) | (_) | (_| |           ")
        print("\____/ \__,_|\__,_|   \_\                                                                 /_/    \____/\___/ \___/ \__,_|           ")
        print("                                                                                                                                    ")
        print("                                                                                                                                    ")
        print("                      __                               _____                       __    _____                                      ")
        print("                     /  |                             |  ___|                     /  |  |  _  |                                     ")
        print("                     `| |                             |___ \                      `| |  | |/' |                                     ")
        print("                      | |                                 \ \                      | |  |  /| |                                     ")
        print("                     _| |_                            /\__/ /                     _| |_ \ |_/ /                                     ")
        print("                     \___/                            \____/                      \___/  \___/                                      ")
        print("                                                                                                                                    ")

#Fungsi methodSVD adalah fungsi utama yang nantinya akan dipanggil ketika user memasukkan angka 1
def methodSVD():
    print("\nSingle Value Decomposition Compression Program")
    print("===============================================\n")
    
    #Input Path image, asumsi selalu benar
    path = input("Masukkan path image : ")
    
    draw(2)
    
    #Input tingkat resolusi gambar, semakin kecil angka semakin besar hasil kompresi dan tentunya gambar semakin bures
    tingkatResolusi = int(input("Masukkan skala kompressi yang diinginkan : "))
    loadImage = cv2.imread(path)
    Image = np.array(loadImage, dtype=np.float64) #Ubah image menjadi kubus array

    #Dekomposisi matriks kubus RGB menjadi matriks 2D
    MatrixR = Image[:,:,0]
    MatrixG = Image[:,:,1]
    MatrixB = Image[:,:,2]

    # Kolom dan Baris masing2 matriks RGB
    mR = np.shape(MatrixR)[0]  
    nR = np.shape(MatrixR)[1]
    mG = np.shape(MatrixG)[0]  
    nG = np.shape(MatrixG)[1]
    mB = np.shape(MatrixB)[0]  
    nB = np.shape(MatrixB)[1]
    
    print("\nCompressing Image...")
    
    # 2 Model Matriks SVD, vertical dan horizontal, bedanya ada di hasil perkalian matriks transposenya AtA dan AAt
    if (mR > nR):
        AtA_R = np.dot(MatrixR.transpose(),MatrixR)
        sigma_R, v_R = np.linalg.eig(AtA_R)
        u_R = getU(MatrixR,v_R, sigma_R)
        
    else:
        AAt_R = np.dot(MatrixR, MatrixR.transpose())
        sigma_R, u_R = np.linalg.eig(AAt_R)
        v_R = getV(MatrixR,u_R, sigma_R)

    PictR = creatingImgCompress(mR,nR,sigma_R, u_R, v_R, tingkatResolusi)
        
    if (mG > nG):
        AtA_G = np.dot(MatrixG.transpose(),MatrixG)
        sigma_G, v_G = np.linalg.eig(AtA_G)
        u_G = getU(MatrixG,v_G, sigma_G)
    else:
        AAt_G = np.dot(MatrixG, MatrixG.transpose())
        sigma_G, u_G = np.linalg.eig(AAt_G)
        v_G = getV(MatrixG,u_G, sigma_G)
    
    PictG = creatingImgCompress(mG,nG,sigma_G, u_G, v_G, tingkatResolusi)
    
    if (mB > nB):
        AtA_B = np.dot(MatrixB.transpose(),MatrixB)
        sigma_B, v_B = np.linalg.eig(AtA_B)
        u_B = getU(MatrixB,v_B, sigma_B)
    else:
        AAt_B = np.dot(MatrixB, MatrixB.transpose())
        sigma_B, u_B = np.linalg.eig(AAt_B)
        v_B = getV(MatrixB,u_B, sigma_B)

    PictB = creatingImgCompress(mB,nB,sigma_B, u_B, v_B, tingkatResolusi)

    rimg = np.zeros(Image.shape)
        
    # Reconstruct 3 Matriks RGB menjadi 1 dan menjadi image
    rimg[:,:,0] = PictR
    rimg[:,:,1] = PictG
    rimg[:,:,2] = PictB

    # Push Image Compress to Folder
    cv2.imwrite('FolderOut/compressedSVD.jpg'.format(tingkatResolusi), rimg)

    ori = os.path.getsize(path)
    palsu = os.path.getsize('FolderOut/compressedSVD.jpg')

    # Hasil Compressi diambil dari size of pict original dan pict of compressed, size yang diambil dapat dilihat dengan klik kanan properties bukan dari GUI size dilayarnya langsung(karena itu hasil pembulatan)
    Persentase = 100-(palsu * 100/ori)
    if (Persentase < 0):
        print("Skala kompresi terlalu besar, sehingga Image menjadi uncompressed")
        os.remove('FolderOut/compressedSVD.jpg')
    else :
        print("Persentase Kompressi : ",100-(palsu * 100/ori),"%")
        print("\nGambar Berhasil Tersimpan")

    # Program Exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     
# Panduan bantuan Pengkoreksian :
# Method Huffman Code
# 1. Ubah Image dalam bentuk integer
# 2. Pisahkan matriks kubus RGB menjadi 2D seperti SVD
# 3. Cari frequensi integer dalam matriks simpan dalam suatu array
# 4. Generating Binary tree berdasarkan frequensi
# 5. Kompress Binary tree dengan algoritma compress image Huffman Code, https://www.youtube.com/watch?v=acEaM2W-Mfw
# 6. Balikin lagi binary code yang udah di compress jadi suatu matriks RGB
# 7. Balikin lagi matriks RGB jadi Image

#Fungsi untuk mendapatkan frequensi yang muncul dari integer Image yang telah di bentuk
def frequensiDataMatrix(dataMatrix):
    MatrixAngka1 = []
    MatrixAngka2 = []
    for i in dataMatrix:
        if i not in MatrixAngka1:
            frekuensiAngka = dataMatrix.count(i) 
            MatrixAngka1.append(frekuensiAngka)
            MatrixAngka1.append(i)
            MatrixAngka2.append(i)
    return MatrixAngka1,MatrixAngka2

#Fungsi untuk mensorting frequensi
def sortingFrekuensiToTree(MatrixAngka1):
    arrayNode = []
    while len(MatrixAngka1) > 0:
        arrayNode.append(MatrixAngka1[0:2])
        MatrixAngka1 = MatrixAngka1[2:]
    arrayNode.sort()
    
    treeKodeHuffman = []
    treeKodeHuffman.append(arrayNode)

    return treeKodeHuffman,arrayNode

def gabungNode(nodes,treeKodeHuffman):
    nodeTree = []
    posisi = 0
    
    if len(nodes) > 1:
        nodes.sort()
        nodes[posisi].append("1")                       # assigning values 1 and 0
        nodes[posisi+1].append("0")
        
        node1gabungan = (nodes[posisi] [0] + nodes[posisi+1] [0])
        node2gabungan = (nodes[posisi] [1] + nodes[posisi+1] [1])  # combining the nodes to generate pathways
        
        nodeTree.append(node1gabungan)
        nodeTree.append(node2gabungan)
        
        nodeFix=[]
        nodeFix.append(nodeTree)
        nodeFix = nodeFix + nodes[2:]
        nodes = nodeFix
        
        treeKodeHuffman.append(nodes)
        gabungNode(nodes,treeKodeHuffman)
    return treeKodeHuffman

# Fungsi untuk mendapatkan image compressi dari algoritma Huffman
def ImgCompressHuffman(img, matrixImageRGB, m, n):
    #Preprocessing image dulu seperti SVD
    if (m > n):
        matrix = np.dot(matrixImageRGB.transpose(),matrixImageRGB)
        sigmaPict, vPict = np.linalg.eig(matrix)
        uPict = getU(matrixImageRGB,vPict, sigmaPict)
        
    else:
        matrix = np.dot(matrixImageRGB, matrixImageRGB.transpose())
        sigmaPict, uPict = np.linalg.eig(matrix)
        vPict = getV(matrixImageRGB,uPict, sigmaPict)
    
    B = np.zeros((m,n))
    matriksPict = np.zeros((m, n), dtype=np.uint8)
    
    for i in range(5):
        B = np.add(B, sigmaPict[i]*np.outer(uPict[:, i], vPict[:, i]))

    for i in range(len(B)):
        for j in range(len(B[0])):
            if B[i][j] > 255:
                matriksPict[i][j] = 255
            elif B[i][j] < 0:
                matriksPict[i][j] = 0
            else:
                matriksPict[i][j] = B[i][j]
                
    
    #Cari frequensi integer image
    ArrayFrekuensiNode1,ArrayFrekuensiNode2 = frequensiDataMatrix(img)
    
    #Sorting frequensi dan buat jadi tree
    treeKodeHuffman, arrayNode = sortingFrekuensiToTree(ArrayFrekuensiNode1)
    
    MergeNodeTreeHuffman = gabungNode(arrayNode,treeKodeHuffman)
    
    treeKodeHuffman.sort(reverse = True)

    checklist = []
    for level in treeKodeHuffman:
        for node in level:
            if node not in checklist:
                checklist.append(node)
            else:
                level.remove(node)
    
    #Generating binary code
    letter_binary = []
    if len(ArrayFrekuensiNode2) == 1:
        lettercode = [ArrayFrekuensiNode2[0], "0"]
        letter_binary.append(lettercode*len(img))
    else:
        for letter in ArrayFrekuensiNode2:
            code =""
            for node in checklist:
                if len (node)>2 and letter in node[1]:
                    code = code + node[2]
            lettercode =[letter,code]
            letter_binary.append(lettercode)

    bitstring =""
    for character in img:
        for item in letter_binary:
            if character in item:
                bitstring = bitstring + item[1]
    binary ="0b"+bitstring
    
    return binary, matriksPict

#Method ini digunakan sebagai fungsi utama saaat user memasukkan input 2
def methodKodeHuffman():
    print("\nHuffman Code Compression Program")
    print("===============================================\n")
    #Load Path
    path = input("Masukkan path image : ")
    image = np.asarray(Image.open(path),np.uint8)
    loadImage = cv2.imread(path)
    imageLoadArray = np.array(loadImage, dtype=np.float64)
    image = str(image.tolist())
    
    #Pecah Matriks kubus
    MatrixR = imageLoadArray[:,:,0]  # array for R
    MatrixG = imageLoadArray[:,:,1]  # array for G
    MatrixB = imageLoadArray[:,:,2] # array for B

    #Dapetin kolom dan baris
    mR = np.shape(MatrixR)[0]  
    nR = np.shape(MatrixR)[1]
    mG = np.shape(MatrixG)[0]  
    nG = np.shape(MatrixG)[1]
    mB = np.shape(MatrixB)[0]  
    nB = np.shape(MatrixB)[1]
    
    print("\nCompressing Image...\n")
    
    print("Making Binary Tree for Red Color...\n")
    BinerR,PictR = ImgCompressHuffman(image, MatrixR, mR, nR)
    print("Image berhasil disimpan dalam bentuk binary didalam file compressImageBinaryRed.txt\n")
    output = open("FolderOut/compressImageBinaryRed.txt","w+")
    output.write(BinerR)
    
    print("Making Binary Tree for Green Color...\n")
    BinerG,PictG = ImgCompressHuffman(image, MatrixG, mG, nG)
    print("Image berhasil disimpan dalam bentuk binary didalam file compressImageBinaryGreen.txt\n")
    output = open("FolderOut/compressImageBinaryGreen.txt","w+")
    output.write(BinerG)
    
    print("Making Binary Tree for Blue Color...\n")
    BinerB,PictB = ImgCompressHuffman(image, MatrixB, mB, nB)
    print("Image berhasil disimpan dalam bentuk binary didalam file compressImageBinaryBlue.txt\n")
    output = open("FolderOut/compressImageBinaryBlue.txt","w+")
    output.write(BinerB)

    #Reconstruct Image dari masing2 matriks RGB
    rimg = np.zeros(imageLoadArray.shape)
    rimg[:,:,0] = PictR
    rimg[:,:,1] = PictG
    rimg[:,:,2] = PictB
    
    #Push Image to folder
    cv2.imwrite('FolderOut/compressedHuffman.jpg', rimg)

    ori = os.path.getsize(path)
    palsu = os.path.getsize('FolderOut/compressedHuffman.jpg')
    
    #Persentase Kompresi sama seperti SVD
    print("Persentase Kompressi : ",100-(palsu * 100/ori),"%")
    print("\nGambar Berhasil Tersimpan")

    #Program exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    draw(1)
    method = int(input("Masukkan methode yang ingin digunakan :\n1. Single Value Decomposition\n2. Kode Huffman\nPilih Metode : "))
    if (method == 1):
        start = time.time()
        methodSVD()
        end = time.time()
        print("Waktu eksekusi program adalah",end - start,"second")
    else:
        start = time.time()
        methodKodeHuffman()
        end = time.time()
        print("Waktu eksekusi program adalah",end - start,"second")
        
main()
    

# aircraft-mask-segmentation

Model için kullanılan data :https://www.kaggle.com/datasets/tayyipcanbay/military-aircraft-detection-dataset-yolo-version


Amaç, savaş uçaklarının video üzerinden bulunma olasılıklarının en yüksek olduğu pikselleri kırmızı yapan bir heatmap elde etmek. Bunun için Savaş uçakları YOLO dataseti kullanılacaktır. 2 farklı yöntem ile YOLO datasından maskeler eldildi. Bu yeni data ile Simple CNN, ResNet, EfficientNet, MobileNet, Vit ve 3'lü input adını verdiğim bir model üzerinde eğitim yapıldı. En iyi sonuç U-Net modelinde elde edildi.

https://github.com/user-attachments/assets/6da74329-338d-48aa-b0bc-cc6d80f266f7

##Yöntem-1 

Maskelerin oluşturulmasında, her image için o imagein boyutlarında grayscale formatında siyah bir maske oluşturulur. Bu maskede YOLO etiketinde bulunan merkez koordinatındaki piksel beyaz olacak; merkezden uzaklaştıkça, width ve height değerlerinin ortalamasından elde edilen yarıçap değerine ulaşıncaya kadar, her piksel için gri ton değeri 0'dan 255'e kadar artan bir gradyan oluşturuluyor.
<img width="1032" alt="Ekran Resmi 2024-11-12 15 58 52" src="https://github.com/user-attachments/assets/fd6ddd7d-9bda-4ec8-bb7c-3fc75fbe3649">

##Yöntem-2

ROI-OTSU benzeri bir thresholding algoritması kullanılarak maskeler elde edildi. Bu yöntemde YOLO datasındaki koordinatlar kullanılarak thresholding yapılacak bölge bir elips içine sınırlandırıldı. Bu sınırın dışında kalan bölgelerin piksel değeri 0 yapıldı. Sınırın içindeki hedef için grayscale formatında eşikleme yapıldı. Tüm data genel olarak 2 aşamadan geçti.

##Genel Data Oluştuma

Burda genel olarak tüm maskeler 2 eşik değere göre ayarlanır: mean_circumference* ve mean_intensity*. Mean_circumference elipsin sırındaki pikseller olamsı sebebiyle arka planın piksellerine aittir bu yüzden beyaza yakınsa “Hedef arka plana göre koyudur” yorumu yapılabilir. Bu durumun tam tersi de tam tersi yorumu yapmaya olanak tanır. Hedef arka plana göre koyuysa mean_intensity değerinin altında kalan pikseller beyaz yapılır, hedef arka plana göre açıksa mean_ intensity değerinin üstündeki pikseller beyez yapılır. Yine de istisnai durumlar olabileceğinden (100-120 resimde bir denk gelir) eşik değere göre doğru olması tahmin edilen maskenin tam tersi output_mask_dir2 dizininde oluşturulur. Maskelerin oluşturulması gradyan inişi ile oluşturlan maskelere göre çok daha hızlı hesaplanır ve modelde daha yüksek doğruluğu vardır ama dezavantajı ekstra manuel inceleme gerektirir. 
(*mean_intensity değeri, yarıçapı elipsin yarıçap değerinin yarısı kadar olan bir elipsin ortalama piksel değeridir)
(*mean_circumference değeri, elips içinde kalan en dış piksellerin ortalama değeridir)

<img width="454" alt="image" src="https://github.com/user-attachments/assets/f254f31e-0a1c-4aad-acbb-e9f6edaf5a6c">


Bazı resimler daha ince işlem gerektirir, bu tarz resimler benim datamda 150-200 resimde bir denk geldi. Bu resimler için while döngüsü içerisinde doğru maske seçilir ya da mean_intensity eşik değeri değiştirilir ve tekrar kontrol edilir. 

<img width="454" alt="image" src="https://github.com/user-attachments/assets/181cddf6-d481-4647-80b1-b4d079646822">


Bu maskeler ile eğitilen hafif bir U-Net modeli kullanılarak video segmentasyonu yapıldı. 




##Projeyi Geliştirmek için Kullanılacak  Modüller

Özellikle motorlu araşlar üzerinde kullanılan kameralarda karşılaşılan problemlerden biri de yüksek frekanslı görüntülerin stabilize edilmesidir. Bunun için yapay zeka ve bazı optik çözümler getirilmiştir. TUBİTAK 2204'e sunduğumuz projemizde yapay zeka veya optik teknolojiler kullanılmadan kuyruk yapısı ve MSE (Mean Square Error) kullanarak real-time bir çözüm getirmeyi amaçlamaktayız.




https://github.com/user-attachments/assets/abe715d3-6738-43dc-86c0-3e41ddd5eb14





##Model Eğitimi için Yapılacaklar

###Data Arttırma

Model mimarisinin karmaşıklaşması FPS i düşüreceğinden mimariyi değiştirmek yerine modelin eğitildiği datayı arttırıcağım. Bunun için videodan image elde edip YOLO ile etiketledikten sonra aynı maskeleme yöntemini kullanacağım.

###Model geliştirme

Data büyüklüğü istenen seviye ulaştığında eğitilen modelin ağırlıklarını ve mimarisini kullanarak bir fine tuning ile model geliştirilicek. Ama bu model zaman ilişkisini de kullanan bir model olucak t zamanında alınan input t-1 zamanındaki inputun outputunu da da belli bir oranda etkileyecek. Bu tarz bir model models dizinindeki 3'lü modellerde belirtilmiştir. Bu modelin fine tune da ihtiyacı olan data uzun bir aircaft videosunun karelerinden elde edilecek.








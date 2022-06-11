const caroHolder = new CaroHolder("mycaro", "mycaroInd",
 ["./assets/img/carousel/1.jpg", "./assets/img/carousel/2.jpg", "./assets/img/carousel/3.jpg", "./assets/img/carousel/4.jpg"],
 ["./assets/img/carousel/m1.jpg", "./assets/img/carousel/m2.jpg", "./assets/img/carousel/m3.jpg", "./assets/img/carousel/m4.jpg"],
 ["https://google.com"]);




fillGrid("presents", 8, "./assets/img/homePage", "jpg", []);




const contentHolder = new ContentHolder();
contentHolder.addContent("./assets/img/items/books/1.png", "", "1$");
contentHolder.addContent("./assets/img/items/books/2.png", "", "2$");
contentHolder.addContent("./assets/img/items/books/3.png", "", "3$");
contentHolder.addContent("./assets/img/items/books/4.png", "", "4$");
contentHolder.addContent("./assets/img/items/books/1.png", "", "5$");
contentHolder.addContent("./assets/img/items/books/2.png", "", "6$");


const carousel = new AdaptiveCarousel(contentHolder);

const cont = document.getElementById("karlsson");
cont.appendChild(carousel.element);




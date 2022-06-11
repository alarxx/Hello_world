const contentHolder = new ContentHolder();
contentHolder.addContent("./assets/img/css3.png", "", "book 1");
contentHolder.addContent("./assets/img/html5.png", "", "book 1");
contentHolder.addContent("./assets/img/jquery.png", "", "tesxt");
contentHolder.addContent("./assets/img/sleepy.jpg", "", "asdf");
contentHolder.addContent("./assets/img/test.jpg", "", "sdfg");
contentHolder.addContent("./assets/img/sca.jpg", "", "dfgh");


const carousel = new AdaptiveCarousel(contentHolder);

const cont = document.getElementById("karlsson");
cont.appendChild(carousel.element);

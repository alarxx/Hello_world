(async () => {
  const {credentials} = require(`./config`);

  const mongoose = require('mongoose');
  await mongoose.connect(credentials.dbUri, {useNewUrlParser: true});

  const express = require('express');
  const app = express();

  const cookieParser = require('cookie-parser');
  app.use(cookieParser(credentials.cookieSecret));

  app.use(express.static(__dirname + '/public'));

  const bodyParser = require('body-parser');
  app.use(bodyParser.json());
  app.use(bodyParser.urlencoded({extended: true}));

  const fileUpload = require('express-fileupload');
  app.use(fileUpload());

  const passport = require('passport');
  require('./auth/passport');


  app.use('/', require('./routes/root'));


  const port = process.env.PORT || 3000;
  app.listen(port, ()=>console.log(`port ${port}`));

})();

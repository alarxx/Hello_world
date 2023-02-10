//Паспорт отвечает только за идентификацию и авторизацию, без аутентификации(регистрация)

const passport = require("passport");
const bcrypt = require('bcrypt');

const LocalStrategy = require('passport-local');

const UserModel = require('../models/User');

passport.use(new LocalStrategy(
	{
		// passReqToCallback: true,
		usernameField: "email",
	},
  // async (req, email, password, done) => {
	async (email, password, done) => {
        const user = await UserModel.findOne({email});

        if (user){
          bcrypt.compare(password, user.password,
            (err, res)=>{
              // console.log(err, res);
              if(res){
                return done(null, user);
              }
              else {
                return done(null, false, {message: err});
              }
            }
          );
        }
        else{
            // console.log({message: "No such user"});
            done(null, false, {message: "No such user"});
        }
    }
));


const {credentials} = require(`../config`);

const JwtStrategy = require('passport-jwt').Strategy;
const {ExtractJwt} = require('passport-jwt');

// Проверяет только ACCESS_TOKEN
passport.use(new JwtStrategy(
  {
    // passReqToCallback: true,
    jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
    secretOrKey: credentials.JWT_ACCESS_TOKEN,
  },
  async (jwt_payload, done) => {
    UserModel.findById(jwt_payload.id, (err, user) => {
        if (err) {
            return done(err, false, {message: err});
        }
        if (user) {
            return done(null, user);
        } else {
            return done(null, false, {message: 'No such user'});
        }
    });
  }
));

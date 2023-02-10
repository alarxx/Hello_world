const passport = require("passport");
const LocalStrategy = require('passport-local');

const bcrypt = require('bcrypt');

const UserModel = require('../models/User');

passport.use(new LocalStrategy(
	{
		passReqToCallback: true, 
		usernameField: "username",
	}, 
	async (req, username, password, done) => {
        const user = await UserModel.findOne({username});

        if (user) { //login
            if(bcrypt.compare(password, user.password))
            	return done(null, user);
            else 
            	return done(null, false, {message: 'Password incorrect'});
        }
        else { //create      
            if(!req.body.username || !req.body.email){}  
            const hash = await bcrypt.hash(password, 10);

            const newUser = {
                username: req.body.username,
                email: req.body.email, 
                password: hash,
                type: 'client',
            };

            const created = await UserModel.create(newUser);

            done(null, created);
        }

    }
))

// Зачем нужны эти методы?
passport.serializeUser(function(user, done){
	// console.log('ser', user); // как '_id' становится 'id'?
    done(null, user);
});
passport.deserializeUser(function(id, done){
	const user = {...id};
    done(null, user);
}); 

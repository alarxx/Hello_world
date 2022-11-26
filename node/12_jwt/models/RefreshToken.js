const {Schema, model} = require('mongoose');

var uniqueValidator = require('mongoose-unique-validator'); //for errors

const RefreshTokenSchema = new Schema({
  user_id: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  token: {
    type: String,
    required: true,
  },
  active: {
    type: Boolean,
    default: true
  }
});

RefreshTokenSchema.plugin(uniqueValidator);

const RefreshToken = model('RefreshToken', RefreshTokenSchema);

module.exports = RefreshToken;

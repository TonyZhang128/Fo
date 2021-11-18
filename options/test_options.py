from .base_options import BaseOptions

#test by mix and separate two videos
class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument(
			'--checkpoint', default="/data/zyn/Foley_extracted/ckpt/best/loss_3.0825_best.pth"
		)
		self.parser.add_argument(
			'--output', default='./output'
		)
		self.parser.add_argument(
			'--video', default=""
		)
		self.parser.add_argument(
			'--instrument', default='Acoustic Grand Piano'
		)
		self.parser.add_argument(
			'--control', default=None
		)
		self.parser.add_argument(
			'--only_audio', action="store_true"
		)
		self.parser.add_argument('--model_name', default='music_transformer', type=str)
		# self.parser.add_argument('--ckpt', default='/data/zyn/Foley_extracted/ckpt', type=str)
		self.parser.add_argument('--emb_dim', default=512, type=int, help='embedding dimension')
		self.parser.add_argument('--hid_dim', default=512, type=int) 
		self.parser.add_argument('--num_encoder_layers', default=0, type=int, help='nums of transformer encoder layers')
		self.parser.add_argument('--num_decoder_layers', default=6, type=int, help='nums of transformer decoder layers')
		self.parser.add_argument('--rpr', default=True, type=bool, help='relative position representation')
		self.parser.add_argument('--rnn', default=None, help='use rnn or not')
		self.parser.add_argument('--decoder_max_seq', default=512, type=int, help='decoder dict max length')
		self.parser.add_argument('--pose_layout', default='body65', type=str)
		self.parser.add_argument('--pose_net_layers', default=10, type=int) 
		self.parser.add_argument('--streams', default=None, help='???')
		self.mode = "test"

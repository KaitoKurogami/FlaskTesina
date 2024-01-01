from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, widgets,SelectMultipleField,IntegerField,FormField, DecimalField
from wtforms.validators import InputRequired, DataRequired
import decimal

class BetterDecimalField(DecimalField):
    """
    Very similar to WTForms DecimalField, except with the option of rounding
    the data always.
    """
    def __init__(self, label=None, validators=None, places=2, rounding=None,
                 round_always=False, **kwargs):
        super(BetterDecimalField, self).__init__(
            label=label, validators=validators, places=places, rounding=
            rounding, **kwargs)
        self.round_always = round_always

    def process_formdata(self, valuelist):
        if valuelist:
            try:
                self.data = decimal.Decimal(valuelist[0])
                if self.round_always and hasattr(self.data, 'quantize'):
                    exp = decimal.Decimal('.1') ** self.places
                    if self.rounding is None:
                        quantized = self.data.quantize(exp)
                    else:
                        quantized = self.data.quantize(
                            exp, rounding=self.rounding)
                    self.data = quantized
            except (decimal.InvalidOperation, ValueError):
                self.data = None
                raise ValueError(self.gettext('Not a valid decimal value'))

class UploadFileForm(FlaskForm):
    file = FileField(label="File", validators=[InputRequired()])
    #submit = SubmitField("Confirm File")

class SHAPForm(FlaskForm):
    SHAP_evals=IntegerField("SHAP_evals",description="1000 por defecto",default=1000, validators=[DataRequired()],render_kw={'disabled':'true'})
    SHAP_batch_size=IntegerField("Shap_bnatch_size", validators=[DataRequired()],default=50,render_kw={'disabled':'true'})
    
class LIMEForm(FlaskForm):
    LIME_perturbations=IntegerField("LIME_perturbations",default=500, validators=[DataRequired()],render_kw={'disabled':'true'})
    LIME_kernel_size=BetterDecimalField(label="LIME_kernel_size", validators=[DataRequired()],round_always=True,default=2.5,render_kw={'disabled':'true'})
    LIME_max_dist=BetterDecimalField(label="LIME_max_dist", validators=[DataRequired()],round_always=True,default=28,render_kw={'disabled':'true'})
    LIME_ratio=BetterDecimalField(label="LIME_ratio", validators=[DataRequired()],round_always=True,default=0.3,render_kw={'disabled':'true'})

class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()

class FullForm(FlaskForm):
    string_of_nets = ['VGG16\r\nResNet50\r\n']
    list_of_nets = string_of_nets[0].split()
    # create a list of value/description tuples
    nets = [(x, x) for x in list_of_nets]
    nets = MultiCheckboxField(label='nets', choices=nets,render_kw={'class':'checkboxNets'})
    file = FormField(UploadFileForm)
    shap = FormField(SHAPForm)
    lime = FormField(LIMEForm)
    submit=SubmitField(label="SubmitButton",render_kw={'disabled':'true'})
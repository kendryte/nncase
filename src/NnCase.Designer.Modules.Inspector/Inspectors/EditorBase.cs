using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;
using NnCase.Designer.Services;
using ReactiveUI;
using Splat;

namespace NnCase.Designer.Modules.Inspector.Inspectors
{
    public abstract class EditorBase<TValue> : InspectorBase, IEditor, IDisposable
    {
        private BoundPropertyDescriptor _boundPropertyDescriptor;
        private IShell _shell;

        public EditorBase()
        {
            _shell = Locator.Current.GetService<IShell>();
            IsUndoEnabled = true;
        }

        public bool IsUndoEnabled
        {
            get;
            set;
        }

        public bool CanReset
        {
            get
            {
                if (IsReadOnly)
                    return false;

                return BoundPropertyDescriptor.PropertyDescriptor.CanResetValue(BoundPropertyDescriptor.PropertyOwner);
            }
        }

        public void Reset()
        {
            if (CanReset)
            {
                BoundPropertyDescriptor.PropertyDescriptor.ResetValue(BoundPropertyDescriptor.PropertyOwner);
            }
        }

        public override string Name
        {
            get { return BoundPropertyDescriptor.PropertyDescriptor.DisplayName; }
        }

        public string Description
        {
            get
            {
                if (!string.IsNullOrEmpty(BoundPropertyDescriptor.PropertyDescriptor.Description))
                    return BoundPropertyDescriptor.PropertyDescriptor.Description;
                return Name;
            }
        }

        public IValueConverter Converter
        {
            get;
            set;
        }

        public IValueConverter StringConverter
        {
            get;
            set;
        }

        private void CleanupPropertyChanged()
        {
            if (_boundPropertyDescriptor != null)
            {
                if (_boundPropertyDescriptor.PropertyDescriptor.SupportsChangeEvents)
                {
                    _boundPropertyDescriptor.ValueChanged -= OnValueChanged;
                }
                else if (typeof(INotifyPropertyChanged).IsAssignableFrom(_boundPropertyDescriptor.PropertyOwner.GetType()))
                {
                    ((INotifyPropertyChanged)_boundPropertyDescriptor.PropertyOwner).PropertyChanged -= OnPropertyChanged;
                }
            }
        }

        public BoundPropertyDescriptor BoundPropertyDescriptor
        {
            get { return _boundPropertyDescriptor; }
            set
            {
                CleanupPropertyChanged();

                _boundPropertyDescriptor = value;

                if (value.PropertyDescriptor.SupportsChangeEvents)
                {
                    value.ValueChanged += OnValueChanged;
                }
                else if (typeof(INotifyPropertyChanged).IsAssignableFrom(value.PropertyOwner.GetType()))
                {
                    ((INotifyPropertyChanged)value.PropertyOwner).PropertyChanged += OnPropertyChanged;
                }
            }
        }

        public override bool IsReadOnly
        {
            get { return BoundPropertyDescriptor.PropertyDescriptor.IsReadOnly; }
        }

        public bool IsDirty
        {
            get
            {
                DefaultValueAttribute defaultAttribute = BoundPropertyDescriptor.PropertyDescriptor.Attributes.OfType<DefaultValueAttribute>().FirstOrDefault();
                if (defaultAttribute == null)
                    /* Maybe not dirty, but we have no way to know if we don't have a default value */
                    return true;

                return !Equals(defaultAttribute.Value, Value);
            }
        }

        private void OnValueChanged()
        {
            this.RaisePropertyChanged(nameof(Value));
            this.RaisePropertyChanged(nameof(IsDirty));
        }

        private void OnValueChanged(object sender, EventArgs e)
        {
            OnValueChanged();
        }

        private void OnPropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName.Equals(BoundPropertyDescriptor.PropertyDescriptor.Name))
                OnValueChanged();
        }

        public TValue Value
        {
            get
            {
                if (!typeof(TValue).IsAssignableFrom(BoundPropertyDescriptor.PropertyDescriptor.PropertyType))
                {
                    if (Converter == null)
                        throw new InvalidCastException("editor property value does not match editor type and no converter specified");

                    return (TValue)Converter.Convert(RawValue, typeof(TValue), null, CultureInfo.CurrentCulture);
                }

                return (TValue)RawValue;
            }

            set
            {
                if (Equals(Value, value))
                    return;

                object newValue = value;
                if (!typeof(TValue).IsAssignableFrom(BoundPropertyDescriptor.PropertyDescriptor.PropertyType))
                {
                    if (Converter == null)
                        throw new InvalidCastException("editor property value does not match editor type and no converter specified");

                    newValue = Converter.ConvertBack(value, BoundPropertyDescriptor.PropertyDescriptor.PropertyType, null, CultureInfo.CurrentCulture);
                }

                /* Only notify of property change once */
                using (this.SuppressChangeNotifications())
                {
                    var item = _shell.ActiveLayoutItem;
                    RawValue = newValue;
                }

                OnValueChanged();
            }
        }

        protected object RawValue
        {
            get { return BoundPropertyDescriptor.Value; }
            set { BoundPropertyDescriptor.Value = value; }
        }

        public virtual void Dispose()
        {
            CleanupPropertyChanged();
        }
    }
}

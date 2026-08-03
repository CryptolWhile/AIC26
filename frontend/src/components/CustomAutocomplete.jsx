import Autocomplete from '@mui/material/Autocomplete';
import CustomTextField from './CustomTextField';

const CustomAutocomplete = ({label, value, onChange, width, ...props}) => {
    return (
        <Autocomplete disablePortal {...props} sx={{ width: width }} value={value} onChange={onChange}
            renderInput={(params) => <CustomTextField {...params} label={label} sx={{ width: '100%' }} />}
        />
    );
};

export default CustomAutocomplete;
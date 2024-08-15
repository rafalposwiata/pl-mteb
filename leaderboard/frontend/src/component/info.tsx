import React from "react";


export interface InfoIconProps {
    text: string;
    warning?: boolean;
}

export class InfoIcon extends React.Component<InfoIconProps, any> {

    render() {
        const text= this.props.text;
        const val = this.props.warning ? "!" : "?";
        let classes = "circled-icon" + (this.props.warning ? " warning" : "")
        return <span className={classes} data-tooltip-id="table-tooltip" data-tooltip-content={text}>{val}</span>
    }
}
import React, {useEffect} from 'react';

import SiteLayout from './SiteLayout.js'


export default function Main(){
	return (
		<SiteLayout menu={<p>Menu</p>}>
			<>
				<Callout>Callout</Callout>
				<h1>Contents</h1>
				<p>This is the main part of the example</p>
			</>
		</SiteLayout>
	);
}
